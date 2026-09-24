import shutil
from collections import Counter
from hashlib import sha256
from pathlib import Path

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)
from qdrant_client import QdrantClient, models
from transformers import AutoTokenizer


def _content_hash(text: str) -> str:
    return sha256(
        text.encode("utf-8")
    ).hexdigest()


def _stored_hashes(
    client: QdrantClient,
    collection_name: str,
) -> Counter[str]:
    hashes: list[str] = []
    offset = None

    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in points:
            if not point.payload:
                continue

            content_hash = point.payload.get(
                "content_hash"
            )

            if isinstance(content_hash, str):
                hashes.append(
                    content_hash
                )

        if offset is None:
            break

    return Counter(hashes)


def _verify_collection(
    client: QdrantClient,
    collection_name: str,
    expected_hashes: Counter[str],
) -> None:
    expected_count = sum(
        expected_hashes.values()
    )

    actual_count = client.count(
        collection_name=collection_name,
        exact=True,
    ).count

    if actual_count != expected_count:
        raise RuntimeError(
            "RAG index verification failed: "
            f"expected {expected_count} chunks, "
            f"found {actual_count}."
        )

    actual_hashes = _stored_hashes(
        client,
        collection_name,
    )

    if actual_hashes != expected_hashes:
        raise RuntimeError(
            "RAG index verification failed: "
            "stored content does not match "
            "imported content."
        )


def ingest_knowledge(
    config: dict,
    customer_id: str,
    customers_directory: Path = Path(
        "customers"
    ),
) -> None:
    rag_config = config["rag"]

    knowledge_directory = (
        customers_directory
        / customer_id
        / "knowledge"
    )

    if not knowledge_directory.is_dir():
        raise FileNotFoundError(
            "Knowledge directory not found: "
            f"{knowledge_directory}"
        )

    files = [
        path
        for path
        in knowledge_directory.rglob("*")
        if path.is_file()
    ]

    if not files:
        raise ValueError(
            "No knowledge documents found in: "
            f"{knowledge_directory}"
        )

    embedding_model = rag_config[
        "embedding_model"
    ]

    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(
            embedding_model
        ),
    )

    chunker = HybridChunker(
        tokenizer=tokenizer,
    )

    converter = DocumentConverter()

    documents: list[str] = []
    metadata: list[dict] = []

    for result in converter.convert_all(
        files
    ):
        source = result.input.file.name

        for chunk in chunker.chunk(
            result.document
        ):
            text = chunk.text.strip()

            if not text:
                continue

            print("\n" + "=" * 80)
            print(f"SOURCE: {source}")
            print("CHUNK:")
            print(text)
            print("=" * 80)

            embedding_text = chunker.contextualize(
                chunk=chunk
            ).strip()

            if not embedding_text:
                embedding_text = text

            print("\n" + "=" * 80)
            print(f"SOURCE: {source}")
            print("\nTEXT:")
            print(text)
            print("\nEMBEDDING TEXT:")
            print(embedding_text)
            print("=" * 80)

            content_hash = _content_hash(
                text
            )

            documents.append(
                embedding_text
            )

            metadata.append(
                {
                    "text": text,
                    "embedding_text": embedding_text,
                    "content_hash": content_hash,
                    "customer_id": customer_id,
                    "source": source,
                    "docling": (
                        chunk.meta.export_json_dict()
                    ),
                }
            )

    if not documents:
        raise ValueError(
            "Knowledge documents produced "
            "no chunks."
        )

    qdrant_path = Path(
        rag_config["qdrant_path"]
    )

    collection_name = rag_config[
        "collection"
    ]
    sparse_model = rag_config[
        "sparse_model"
    ]

    dense_vector_name = "dense"
    sparse_vector_name = "sparse"

    expected_hashes = Counter(
        payload["content_hash"]
        for payload in metadata
    )

    # Qdrant Local + Windows:
    # Do not delete/recreate a collection
    # inside the same local client.
    #
    # Start from a completely fresh local
    # storage directory instead.
    if qdrant_path.exists():
        try:
            shutil.rmtree(
                qdrant_path
            )
        except OSError as exc:
            raise RuntimeError(
                "Could not reset local Qdrant "
                "storage. Make sure start.cmd "
                "is not running."
            ) from exc

    client = QdrantClient(
        path=str(qdrant_path),
    )

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                dense_vector_name: (
                    models.VectorParams(
                        size=(
                            client.get_embedding_size(
                                embedding_model
                            )
                        ),
                        distance=(
                            models.Distance.COSINE
                        ),
                    )
                )
            },
            sparse_vectors_config={
                sparse_vector_name: (
                    models.SparseVectorParams()
                )
            },
        )

        empty_count = client.count(
            collection_name=collection_name,
            exact=True,
        ).count

        if empty_count != 0:
            raise RuntimeError(
                "RAG collection rebuild failed: "
                "fresh collection contains "
                f"{empty_count} points."
            )

        points = [
            models.PointStruct(
                id=index,
                vector={
                    dense_vector_name: (
                        models.Document(
                            text=text,
                            model=embedding_model,
                        )
                    ),
                    sparse_vector_name: (
                        models.Document(
                            text=text,
                            model=sparse_model,
                        )
                    ),
                },
                payload=payload,
            )
            for index, (
                text,
                payload,
            ) in enumerate(
                zip(
                    documents,
                    metadata,
                )
            )
        ]

        client.upload_points(
            collection_name=collection_name,
            points=points,
        )

        _verify_collection(
            client,
            collection_name,
            expected_hashes,
        )

    finally:
        client.close()

    # Reopen from disk and verify again.
    # This is the exact persisted state
    # start.cmd will see afterwards.
    client = QdrantClient(
        path=str(qdrant_path),
    )

    try:
        _verify_collection(
            client,
            collection_name,
            expected_hashes,
        )

    finally:
        client.close()

    print(
        f"Indexed {len(documents)} chunks "
        f"from {len(files)} document(s)."
    )
    print(
        "Knowledge index verified."
    )


if __name__ == "__main__":
    from local_phone_ai_v2.config.loader import (
        load_config,
    )
    from local_phone_ai_v2.config.validate import (
        validate_config,
    )

    config = load_config(
        Path("config.yaml")
    )
    validate_config(
        config
    )

    ingest_knowledge(
        config=config,
        customer_id="test_customer",
    )
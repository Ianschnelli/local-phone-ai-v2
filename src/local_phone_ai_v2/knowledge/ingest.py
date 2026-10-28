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

from local_phone_ai_v2.knowledge.embedder import DenseEmbedder
from local_phone_ai_v2.knowledge.enrichment import (
    ChunkEnricher,
    build_enriched_embedding_text,
)
from local_phone_ai_v2.knowledge.index_manifest import (
    build_index_fingerprint,
    write_index_manifest,
)


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

    initial_fingerprint = build_index_fingerprint(
        config,
        knowledge_directory,
    )

    embedder = DenseEmbedder(rag_config)

    # Abschnittsgröße unabhängig vom Embedding-Modell, damit ein Modellwechsel
    # die Abschnitte nicht verändert (E-015). Standard: bisheriges Modell.
    chunk_tokenizer = rag_config.get(
        "chunk_tokenizer",
        rag_config["embedding_model"],
    )

    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(
            chunk_tokenizer,
            local_files_only=True,
        ),
    )

    chunker = HybridChunker(
        tokenizer=tokenizer,
    )

    converter = DocumentConverter()

    # Optionale Anreicherung mit Anruferformulierungen (E-016), braucht LM Studio.
    enrichment_config = rag_config.get("enrichment") or {}
    enricher = None

    if enrichment_config.get("enabled"):
        enricher = ChunkEnricher(
            llm_config=config["llm"],
            cache_path=(
                Path(enrichment_config.get("cache_dir", "data/enrichment"))
                / f"{customer_id}.json"
            ),
        )

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

            enrichment = None

            if enricher is not None:
                headings = chunk.meta.headings or [source]
                enrichment = enricher.enrich(
                    content_hash=_content_hash(text),
                    title=headings[0],
                    heading=headings[-1],
                    text=text,
                )
                embedding_text = build_enriched_embedding_text(
                    embedding_text,
                    enrichment,
                )

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
                    "enrichment": enrichment,
                    "docling": (
                        chunk.meta.export_json_dict()
                    ),
                }
            )

    if enricher is not None:
        enricher.save()

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

    # Build a new index without touching the active one.
    staging_path = qdrant_path.with_name(
        qdrant_path.name + ".__building"
    )
    backup_path = qdrant_path.with_name(
        qdrant_path.name + ".__backup"
    )

    # Never overwrite a backup left by a failed rebuild.
    if backup_path.exists():
        raise RuntimeError(
            f"Existing Qdrant backup: {backup_path}. "
            "Inspect it before rebuilding."
        )

    # An incomplete staging index can be rebuilt.
    if staging_path.exists():
        shutil.rmtree(staging_path)

    client = QdrantClient(
        path=str(staging_path),
    )

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                dense_vector_name: (
                    models.VectorParams(
                        size=(
                            embedder.size(client)
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
                    dense_vector_name: dense_vector,
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
                dense_vector,
            ) in enumerate(
                zip(
                    documents,
                    metadata,
                    embedder.document_vectors(documents),
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

    # Reopen the staged index from disk and verify it.
    client = QdrantClient(
        path=str(staging_path),
    )

    try:
        _verify_collection(
            client,
            collection_name,
            expected_hashes,
        )

    finally:
        client.close()

    # Ensure source documents did not change during indexing.
    final_fingerprint = build_index_fingerprint(
        config,
        knowledge_directory,
    )

    if final_fingerprint != initial_fingerprint:
        raise RuntimeError(
            "Knowledge documents changed during indexing. "
            "The existing Qdrant index remains untouched."
        )

    # Store the manifest inside the verified staging index.
    write_index_manifest(
        staging_path,
        final_fingerprint,
    )

    # Activate the verified index.
    # All Qdrant clients must be closed.
    had_active_index = qdrant_path.exists()

    if had_active_index:
        try:
            qdrant_path.rename(backup_path)
        except OSError as exc:
            raise RuntimeError(
                "Could not back up the active Qdrant index. "
                "Make sure start.cmd is not running."
            ) from exc

    try:
        staging_path.rename(qdrant_path)
    except OSError:
        # Restore the original index if activation fails.
        if had_active_index:
            backup_path.rename(qdrant_path)
        raise

    # The new index is active; clean up the old backup.
    if had_active_index:
        try:
            shutil.rmtree(backup_path)
        except OSError as exc:
            print(
                f"WARNING: Old Qdrant backup remains "
                f"at {backup_path}: {exc}"
            )

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

    customer_dir = Path(config["customer"]["facts_path"]).parent

    ingest_knowledge(
        config=config,
        customer_id=customer_dir.name,
        customers_directory=customer_dir.parent,
    )
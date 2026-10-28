import json
import urllib.error
import urllib.request
from pathlib import Path

from fastembed import (
    SparseTextEmbedding,
    TextEmbedding,
)
from faster_whisper.utils import download_model
from huggingface_hub import hf_hub_download
from piper.download_voices import download_voice
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder
from transformers import AutoTokenizer

from local_phone_ai_v2.config.loader import load_config
from local_phone_ai_v2.config.validate import validate_config
from local_phone_ai_v2.knowledge.embedder import DenseEmbedder
from local_phone_ai_v2.knowledge.ingest import (
    ingest_knowledge,
)
from local_phone_ai_v2.knowledge.index_manifest import (
    build_index_fingerprint,
    index_is_current,
)

KOKORO_REPO = "Godelaune/Kokoro-82M-ONNX-German-Martin"

KOKORO_FILES = (
    "kokoro-martin.onnx",
    "voices-martin.npz",
)


def ensure_lm_studio_model(config: dict) -> None:
    base_url = config["base_url"].rstrip("/")
    model = config["model"]

    try:
        with urllib.request.urlopen(
            f"{base_url}/models",
            timeout=5,
        ) as response:
            payload = json.load(response)

    except (
        urllib.error.URLError,
        TimeoutError,
    ) as exc:
        raise RuntimeError(
            "LM Studio is not reachable. "
            f"Expected API at {base_url}"
        ) from exc

    available_models = {
        item["id"]
        for item in payload.get("data", [])
        if "id" in item
    }

    if model not in available_models:
        raise RuntimeError(
            f"LM Studio model missing: {model}. "
            f"Available models: {sorted(available_models)}"
        )

    print(
        f"LM Studio model OK: {model}"
    )


def ensure_whisper_model(model: str) -> None:
    print(
        f"Checking Whisper model: {model}"
    )

    download_model(model)

    print(
        f"Whisper model OK: {model}"
    )


def ensure_kokoro_model() -> None:
    model_dir = Path(
        "models/kokoro_martin"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for filename in KOKORO_FILES:
        target = model_dir / filename

        if target.exists():
            print(
                f"Kokoro file OK: {filename}"
            )
            continue

        print(
            f"Downloading Kokoro file: {filename}"
        )

        hf_hub_download(
            repo_id=KOKORO_REPO,
            filename=filename,
            local_dir=model_dir,
        )


def ensure_piper_model(model: str) -> None:
    model_dir = Path(
        "models/piper"
    )

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        model_dir / f"{model}.onnx"
    )

    config_path = (
        model_dir / f"{model}.onnx.json"
    )

    if (
        model_path.exists()
        and config_path.exists()
    ):
        print(
            f"Piper model OK: {model}"
        )
        return

    print(
        f"Downloading Piper model: {model}"
    )

    download_voice(
        model,
        model_dir,
    )

    print(
        f"Piper model OK: {model}"
    )


def ensure_tts_model(config: dict) -> None:
    provider = config["provider"]

    if provider == "kokoro":
        ensure_kokoro_model()
        return

    if provider == "piper":
        ensure_piper_model(
            config["model"]
        )
        return

    raise ValueError(
        f"Unsupported TTS provider: {provider}"
    )


def ensure_rag_models(
    config: dict,
) -> None:
    rag = config["rag"]

    dense_model = (
        rag["embedding_model"]
    )

    sparse_model = (
        rag["sparse_model"]
    )

    print(
        f"Checking dense embedding model: "
        f"{dense_model}"
    )

    if rag.get("embedding_backend", "fastembed") == "sentence_transformers":
        # lädt das Modell beim ersten Mal herunter (E-015)
        DenseEmbedder(rag, local_files_only=False).query_vector("setup")
    else:
        dense = TextEmbedding(
            model_name=dense_model,
        )

        list(
            dense.embed(
                ["setup"]
            )
        )

    # Tokenizer für die Abschnittsgröße beim Einlesen
    AutoTokenizer.from_pretrained(
        rag.get("chunk_tokenizer", dense_model)
    )

    print(
        f"Dense embedding model OK: "
        f"{dense_model}"
    )

    print(
        f"Checking sparse embedding model: "
        f"{sparse_model}"
    )

    sparse = SparseTextEmbedding(
        model_name=sparse_model,
    )

    list(
        sparse.embed(
            ["setup"]
        )
    )

    print(
        f"Sparse embedding model OK: "
        f"{sparse_model}"
    )


def ensure_reranker(
    config: dict,
) -> None:
    reranker = (
        config["rag"]["reranker"]
    )

    model = reranker["model"]

    print(
        f"Checking reranker model: {model}"
    )

    CrossEncoder(
        model,
        device="cpu",
    )

    print(
        f"Reranker model OK: {model}"
    )

def ensure_rag_index(config: dict) -> None:
    rag = config["rag"]

    qdrant_path = Path(rag["qdrant_path"])
    collection = rag["collection"]

    facts_path = Path(config["customer"]["facts_path"])
    customer_id = facts_path.parent.name
    customers_directory = facts_path.parent.parent
    knowledge_directory = facts_path.parent / "knowledge"

    expected_fingerprint = build_index_fingerprint(
        config,
        knowledge_directory,
    )

    if (
        qdrant_path.exists()
        and index_is_current(
            qdrant_path,
            expected_fingerprint,
        )
    ):
        client = QdrantClient(path=str(qdrant_path))

        try:
            if client.collection_exists(collection):
                count = client.count(
                    collection_name=collection,
                    exact=True,
                ).count

                if count > 0:
                    print(
                        f"RAG index OK: "
                        f"{collection} ({count} chunks)"
                    )
                    return

        finally:
            client.close()

    print(
        "RAG index missing, empty or outdated. "
        "Rebuilding..."
    )

    ingest_knowledge(
        config=config,
        customer_id=customer_id,
        customers_directory=customers_directory,
    )

def main() -> None:
    config = load_config()

    validate_config(config)

    print(
        "Local Phone AI V2 setup"
    )

    print(
        "Config OK"
    )

    ensure_lm_studio_model(
        config["llm"]
    )

    ensure_whisper_model(
        config["stt"]["model"]
    )

    ensure_tts_model(
        config["tts"]
    )

    ensure_rag_models(
        config
    )

    ensure_reranker(
        config
    )
    ensure_rag_index(
        config
    )

    print(
        "Setup complete"
    )


if __name__ == "__main__":
    main()
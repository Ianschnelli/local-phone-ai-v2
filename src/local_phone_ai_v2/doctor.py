import subprocess
import ctranslate2
from pathlib import Path

from local_phone_ai_v2.config.loader import load_config
from local_phone_ai_v2.config.validate import validate_config
from local_phone_ai_v2.services.stt import create_stt

from fastembed import (
    SparseTextEmbedding,
    TextEmbedding,
)
from sentence_transformers import CrossEncoder

from qdrant_client import QdrantClient

from local_phone_ai_v2.services.turn import (
    create_turn_analyzer,
)

def check_config(config: dict) -> None:
    validate_config(config)
    print("[OK] Config")


def check_ollama(model: str) -> None:
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        check=True,
    )

    if model not in result.stdout:
        raise RuntimeError(f"Ollama model missing: {model}")

    print(f"[OK] Ollama model: {model}")


def check_stt(config: dict) -> None:
    create_stt(config)
    print(
        f"[OK] STT: {config['model']} on {config['device']}"
    )

def check_turn(config: dict) -> None:
    if config["turn"]["provider"] != "smart_turn":
        return

    create_turn_analyzer(
        config["audio"]["processing_sample_rate"]
    )

    print("[OK] Smart Turn")

def check_customer_files(config: dict) -> None:
    facts_path = Path(
        config["customer"]["facts_path"]
    )

    customer_dir = facts_path.parent
    customer_path = (
        customer_dir / "customer.yaml"
    )

    if not customer_path.is_file():
        raise RuntimeError(
            f"Customer config missing: "
            f"{customer_path}"
        )

    if not facts_path.is_file():
        raise RuntimeError(
            f"Facts file missing: "
            f"{facts_path}"
        )

    print(
        f"[OK] Customer config: "
        f"{customer_path}"
    )
    print(
        f"[OK] Structured facts: "
        f"{facts_path}"
    )

def check_rag_index(config: dict) -> None:
    rag = config["rag"]

    if not rag["enabled"]:
        print("[OK] RAG disabled")
        return

    qdrant_path = Path(
        rag["qdrant_path"]
    )

    if not qdrant_path.exists():
        raise RuntimeError(
            f"Qdrant path missing: "
            f"{qdrant_path}"
        )

    client = QdrantClient(
        path=str(qdrant_path)
    )

    try:
        collection = rag["collection"]

        if not client.collection_exists(
            collection
        ):
            raise RuntimeError(
                f"RAG collection missing: "
                f"{collection}"
            )

        count = client.count(
            collection_name=collection,
            exact=True,
        ).count

        if count < 1:
            raise RuntimeError(
                f"RAG collection is empty: "
                f"{collection}"
            )

        print(
            f"[OK] RAG index: "
            f"{collection} ({count} chunks)"
        )

    finally:
        client.close()

def check_kokoro() -> None:
    model_dir = Path("models/kokoro_martin")

    required_files = (
        "kokoro-martin.onnx",
        "voices-martin.npz",
    )

    for filename in required_files:
        if not (model_dir / filename).exists():
            raise RuntimeError(f"Kokoro file missing: {filename}")

    print("[OK] Kokoro Martin")

def check_piper(config: dict) -> None:
    model_dir = Path("models/piper")
    model = config["model"]

    required_files = (
        f"{model}.onnx",
        f"{model}.onnx.json",
    )

    for filename in required_files:
        if not (model_dir / filename).exists():
            raise RuntimeError(
                f"Piper file missing: {filename}"
            )

    print(f"[OK] Piper: {model}")

def check_tts(config: dict) -> None:
    provider = config["provider"]

    if provider == "kokoro":
        check_kokoro()
        return

    if provider == "piper":
        check_piper(config)
        return

    raise RuntimeError(
        f"Unsupported TTS provider: {provider}"
    )

def check_stt_runtime(config: dict) -> None:
    if config["device"] != "cuda":
        return

    if ctranslate2.get_cuda_device_count() < 1:
        raise RuntimeError(
            "STT is configured for CUDA, but CTranslate2 "
            "cannot access a CUDA-capable GPU."
        )

    print("[OK] CTranslate2 CUDA device")

def check_rag_models(config: dict) -> None:
    rag = config["rag"]

    dense_model = rag["embedding_model"]
    sparse_model = rag["sparse_model"]

    dense = TextEmbedding(
        model_name=dense_model,
    )
    list(
        dense.embed(
            ["doctor"]
        )
    )

    print(
        f"[OK] Dense embedding: "
        f"{dense_model}"
    )

    sparse = SparseTextEmbedding(
        model_name=sparse_model,
    )
    list(
        sparse.embed(
            ["doctor"]
        )
    )

    print(
        f"[OK] Sparse embedding: "
        f"{sparse_model}"
    )

def check_reranker(config: dict) -> None:
    reranker = config["rag"]["reranker"]

    if not reranker["enabled"]:
        print("[OK] Reranker disabled")
        return

    model = reranker["model"]

    CrossEncoder(
        model,
        device="cpu",
        local_files_only=True,
    )

    print(
        f"[OK] Reranker: {model}"
    )

def main() -> None:
    config = load_config()

    print("Local Phone AI V2 doctor")

    check_config(config)
    check_ollama(config["llm"]["model"])
    check_stt_runtime(config["stt"])
    check_stt(config["stt"])

    check_tts(config["tts"])
    check_rag_models(config)
    check_reranker(config)

    check_turn(config)
    check_customer_files(config)
    check_rag_index(config)

    print("[OK] Preflight complete")


if __name__ == "__main__":
    main()
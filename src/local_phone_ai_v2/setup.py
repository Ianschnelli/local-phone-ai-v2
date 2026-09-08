import subprocess
from pathlib import Path

from faster_whisper.utils import download_model
from huggingface_hub import hf_hub_download

from local_phone_ai_v2.config.loader import load_config
from local_phone_ai_v2.config.validate import validate_config

from fastembed import (
    SparseTextEmbedding,
    TextEmbedding,
)
from piper.download_voices import download_voice
from sentence_transformers import CrossEncoder

KOKORO_REPO = "Godelaune/Kokoro-82M-ONNX-German-Martin"
KOKORO_FILES = (
    "kokoro-martin.onnx",
    "voices-martin.npz",
)


def ensure_ollama_model(model: str) -> None:
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        check=True,
    )

    installed_models = {
        line.split()[0]
        for line in result.stdout.splitlines()[1:]
        if line.strip()
    }

    if model in installed_models:
        print(f"Ollama model OK: {model}")
        return

    print(f"Downloading Ollama model: {model}")
    subprocess.run(["ollama", "pull", model], check=True)


def ensure_whisper_model(model: str) -> None:
    print(f"Checking Whisper model: {model}")
    download_model(model)
    print(f"Whisper model OK: {model}")


def ensure_kokoro_model() -> None:
    model_dir = Path("models/kokoro_martin")
    model_dir.mkdir(parents=True, exist_ok=True)

    for filename in KOKORO_FILES:
        target = model_dir / filename

        if target.exists():
            print(f"Kokoro file OK: {filename}")
            continue

        print(f"Downloading Kokoro file: {filename}")

        hf_hub_download(
            repo_id=KOKORO_REPO,
            filename=filename,
            local_dir=model_dir,
        )

def ensure_piper_model(model: str) -> None:
    model_dir = Path("models/piper")
    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = model_dir / f"{model}.onnx"
    config_path = model_dir / f"{model}.onnx.json"

    if model_path.exists() and config_path.exists():
        print(f"Piper model OK: {model}")
        return

    print(f"Downloading Piper model: {model}")

    download_voice(
        model,
        model_dir,
    )

    print(f"Piper model OK: {model}")

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

def ensure_rag_models(config: dict) -> None:
    rag = config["rag"]

    dense_model = rag["embedding_model"]
    sparse_model = rag["sparse_model"]

    print(
        f"Checking dense embedding model: "
        f"{dense_model}"
    )

    dense = TextEmbedding(
        model_name=dense_model,
    )

    list(
        dense.embed(
            ["setup"]
        )
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

def ensure_reranker(config: dict) -> None:
    reranker = config["rag"]["reranker"]

    if not reranker["enabled"]:
        return

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

def main() -> None:
    config = load_config()
    validate_config(config)

    print("Local Phone AI V2 setup")
    print("Config OK")

    ensure_ollama_model(
        config["llm"]["model"]
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

    print("Setup complete")


if __name__ == "__main__":
    main()
import json
from pathlib import Path
from urllib import error, request

import ctranslate2
from fastembed import (
    SparseTextEmbedding,
    TextEmbedding,
)
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

from local_phone_ai_v2.config.loader import load_config
from local_phone_ai_v2.config.validate import validate_config
from local_phone_ai_v2.services.stt import create_stt
from local_phone_ai_v2.services.turn import (
    create_turn_analyzer,
)


def check_config(config: dict) -> None:
    validate_config(config)
    print("[OK] Config")

def _lm_studio_request(
    url: str,
    api_key: str,
    payload: dict | None = None,
    timeout: int = 10,
) -> dict:
    headers = {
        "Content-Type": "application/json",
    }

    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    data = None

    if payload is not None:
        data = json.dumps(
            payload
        ).encode("utf-8")

    req = request.Request(
        url,
        data=data,
        headers=headers,
        method="POST" if payload is not None else "GET",
    )

    try:
        with request.urlopen(
            req,
            timeout=timeout,
        ) as response:
            body = response.read().decode("utf-8")

    except error.HTTPError as exc:
        body = exc.read().decode(
            "utf-8",
            errors="replace",
        )

        raise RuntimeError(
            f"LM Studio request failed with HTTP "
            f"{exc.code}: {body}"
        ) from exc

    except error.URLError as exc:
        raise RuntimeError(
            f"LM Studio is not reachable: {url}"
        ) from exc

    try:
        return json.loads(body)

    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "LM Studio returned invalid JSON."
        ) from exc


def check_lm_studio(config: dict) -> None:
    base_url = config["base_url"].rstrip("/")
    api_key = config["api_key"]
    model = config["model"]

    models_response = _lm_studio_request(
        f"{base_url}/models",
        api_key,
    )

    available_models = {
        item["id"]
        for item in models_response.get("data", [])
        if isinstance(item, dict)
        and "id" in item
    }

    if model not in available_models:
        raise RuntimeError(
            f"LM Studio model missing: {model}. "
            f"Available models: {sorted(available_models)}"
        )

    print(
        f"[OK] LM Studio model: {model}"
    )

    completion = _lm_studio_request(
        f"{base_url}/chat/completions",
        api_key,
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Antworte ausschließlich "
                        "mit dem Wort TEST."
                    ),
                }
            ],
            "temperature": 0,
            "max_tokens": 32,
            "stream": False,
        },
        timeout=120,
    )

    choices = completion.get("choices", [])

    if not choices:
        raise RuntimeError(
            "LM Studio returned no completion choices."
        )

    message = choices[0].get(
        "message",
        {},
    )

    content = message.get("content")

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(
            "LM Studio returned an empty completion."
        )

    print("[OK] LM Studio completion")

    if config.get("thinking") is not False:
        return

    reasoning_content_available = (
        "reasoning_content" in message
    )

    reasoning_content = message.get(
        "reasoning_content"
    )

    completion_details = (
        completion
        .get("usage", {})
        .get("completion_tokens_details", {})
    )

    reasoning_tokens_available = (
        "reasoning_tokens" in completion_details
    )

    reasoning_tokens = completion_details.get(
        "reasoning_tokens"
    )

    if (
        not reasoning_content_available
        and not reasoning_tokens_available
    ):
        raise RuntimeError(
            "Non-Thinking cannot be verified because "
            "LM Studio returned neither reasoning_content "
            "nor reasoning_tokens."
        )

    if reasoning_content:
        raise RuntimeError(
            "Thinking is active in LM Studio. "
            "Disable 'Enable Thinking' for this model."
        )

    if reasoning_tokens not in (None, 0):
        raise RuntimeError(
            "Thinking is active in LM Studio "
            f"(reasoning_tokens={reasoning_tokens}). "
            "Disable 'Enable Thinking' for this model."
        )

    print("[OK] LM Studio Non-Thinking")


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
    check_lm_studio(config["llm"])
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
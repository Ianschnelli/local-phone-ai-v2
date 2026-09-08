import json
from urllib import request


def _post_json(
    url: str,
    payload: dict,
    timeout: int = 120,
) -> None:
    data = json.dumps(
        payload
    ).encode("utf-8")

    req = request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with request.urlopen(
        req,
        timeout=timeout,
    ) as response:
        if response.status != 200:
            raise RuntimeError(
                f"Warmup failed with HTTP {response.status}."
            )

        response.read()


def warmup_qwen(
    config: dict,
) -> None:
    try:
        # Modell in Ollama laden und resident halten.
        _post_json(
            "http://localhost:11434/api/generate",
            {
                "model": config["model"],
                "prompt": "",
                "stream": False,
                "keep_alive": "30m",
            },
        )

        # Gleichen OpenAI-kompatiblen API-Pfad
        # kurz anstoßen, den Pipecat später nutzt.
        payload = {
            "model": config["model"],
            "messages": [
                {
                    "role": "user",
                    "content": "Bereit.",
                }
            ],
            "stream": False,
            "temperature": config["temperature"],
            "top_p": config["top_p"],
            "max_tokens": 1,
        }

        if config.get("thinking") is False:
            payload["reasoning_effort"] = "none"

        _post_json(
            "http://localhost:11434/v1/chat/completions",
            payload,
        )

    except Exception as exc:
        raise RuntimeError(
            "Qwen warmup failed."
        ) from exc
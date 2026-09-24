import json
from urllib import error, request


def _post_json(
    url: str,
    payload: dict,
    timeout: int = 120,
) -> dict:
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
            f"LM Studio warmup failed with HTTP "
            f"{exc.code}: {body}"
        ) from exc

    except error.URLError as exc:
        raise RuntimeError(
            f"LM Studio is not reachable: {url}"
        ) from exc

    return json.loads(body)


def warmup_qwen(
    config: dict,
) -> None:
    base_url = config["base_url"].rstrip("/")

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

    try:
        _post_json(
            f"{base_url}/chat/completions",
            payload,
        )

    except Exception as exc:
        raise RuntimeError(
            "Qwen warmup via LM Studio failed."
        ) from exc
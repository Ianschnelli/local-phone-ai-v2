"""LM Studio: Zeit bis zum ersten Token (TTFT), ohne Pipeline und ohne Audio.

Aufruf aus dem Projektverzeichnis:
    .venv\\Scripts\\python.exe eval\\llm_latencyALTIN.py [Bezeichnung]

Die Bezeichnung (z. B. "thinking-aus, parallel-1") wird nur mit ausgegeben,
damit sich Läufe nach einer Umstellung in LM Studio vergleichen lassen.
Nutzt den echten Systemprompt der Konfiguration. Ändert nichts.
"""
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

from local_phone_ai_v2.config.loader import load_config
from local_phone_ai_v2.conversation.customer_loader import load_customer_config
from local_phone_ai_v2.conversation.instructions import build_instructions

RUNS = 4


def ttft(base_url: str, model: str, messages: list[dict]) -> tuple[float, float, str]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.5,
        "top_p": 0.8,
        "max_tokens": 60,
        "stream": True,
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first = None
    text = ""
    with urllib.request.urlopen(request, timeout=120) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            delta = json.loads(line[5:])["choices"][0]["delta"].get("content") or ""
            if delta and first is None:
                first = time.perf_counter() - start
            text += delta
    return first or float("nan"), time.perf_counter() - start, text


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "ohne Bezeichnung"
    config = load_config("config.yaml")
    llm = config["llm"]
    base_url = llm["base_url"].rstrip("/")
    model = llm["model"]

    customer = load_customer_config(Path("customers"), "test_customer")
    system = {
        "role": "system",
        "content": build_instructions(customer_instructions=customer["instructions"]),
    }

    cases = {
        "kurz, ohne Systemprompt": [{"role": "user", "content": "Sag Hallo."}],
        "echter Systemprompt": [
            system,
            {"role": "user", "content": "Machen Sie Blutabnahmen?"},
        ],
    }

    print(f"LM Studio TTFT | Modell {model} | Lauf: {label}")
    print(f"Systemprompt: {len(system['content'])} Zeichen\n")
    for name, messages in cases.items():
        values = []
        for _ in range(RUNS):
            first, _total, _text = ttft(base_url, model, messages)
            values.append(first)
        warm = values[1:]  # erster Lauf kann Prompt-Cache füllen
        print(
            f"{name:<26} erstes Token: "
            + " / ".join(f"{v:.2f}s" for v in values)
            + f"  | Median (ohne 1. Lauf) {statistics.median(warm):.2f}s"
        )


if __name__ == "__main__":
    main()

"""Kommt die Gesprächszusammenfassung bei Qwen an? (Befund B-11)

Baut die Anfrage so nach, wie die App sie nach einer Kompaktierung an
LM Studio schickt: system (Regeln) -> Zusammenfassung -> letzte 3 Wortwechsel
-> neue Frage. Der Name des Anrufers steht NUR in der Zusammenfassung.

Varianten: Rolle "developer" (wie die App heute), "system", "user" und eine
Kontrolle ohne Zusammenfassung. Ändert nichts, braucht nur LM Studio.

Aufruf aus dem Projektverzeichnis:
    .venv\\Scripts\\python.exe eval\\memory_summaryALTIN.py
"""
import json
import urllib.error
import urllib.request
from pathlib import Path

from local_phone_ai_v2.config.loader import load_config
from local_phone_ai_v2.conversation.customer_loader import load_customer_config
from local_phone_ai_v2.conversation.instructions import build_instructions
from local_phone_ai_v2.conversation.summary import CallSummary

RUNS = 3

SUMMARY = CallSummary(
    anliegen=["Wiederholungsrezept für Blutdrucktabletten anfragen"],
    genannte_fakten=["Geburtsdatum 12.03.1961"],
    genannte_entitaeten=["Herr Müller"],
    offene_punkte=["Rückruf wegen des Rezepts gewünscht"],
)

# Gleiches Format wie ConversationContextManager.build_summary_message()
SUMMARY_TEXT = (
    "STRUKTURIERTE ZUSAMMENFASSUNG "
    "DES BISHERIGEN GESPRÄCHS:\n"
    f"{SUMMARY.model_dump_json()}"
)

RECENT = [
    {"role": "user", "content": "Wann haben Sie am Donnerstag geöffnet?"},
    {"role": "assistant", "content": "Die Anmeldung ist am Donnerstag von acht bis achtzehn Uhr geöffnet."},
    {"role": "user", "content": "Und die Sprechstunde?"},
    {"role": "assistant", "content": "Die Sprechstunde ist am Donnerstag von acht bis zwölf Uhr."},
    {"role": "user", "content": "Okay, danke."},
    {"role": "assistant", "content": "Gern. Kann ich sonst noch etwas für Sie tun?"},
]

QUESTION = {
    "role": "user",
    "content": "Wie war nochmal mein Name, und weswegen hatte ich angerufen?",
}


def ask(base_url: str, llm: dict, messages: list[dict]) -> str:
    payload = {
        "model": llm["model"],
        "messages": messages,
        "temperature": llm["temperature"],
        "top_p": llm["top_p"],
        "max_tokens": llm["max_tokens"],
        "stream": False,
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.load(response)
    except urllib.error.HTTPError as exc:
        return f"HTTP-FEHLER {exc.code}: {exc.read().decode(errors='replace')[:200]}"
    return (body["choices"][0]["message"].get("content") or "").strip()


def main() -> None:
    config = load_config("config.yaml")
    llm = config["llm"]
    base_url = llm["base_url"].rstrip("/")
    customer = load_customer_config(Path("customers"), "test_customer")
    system = {
        "role": "system",
        "content": build_instructions(customer_instructions=customer["instructions"]),
    }

    variants = {
        'Rolle "developer" (wie die App)': [system, {"role": "developer", "content": SUMMARY_TEXT}],
        'Rolle "system"': [system, {"role": "system", "content": SUMMARY_TEXT}],
        'Rolle "user"': [system, {"role": "user", "content": SUMMARY_TEXT}],
        "Kontrolle: ohne Zusammenfassung": [system],
    }

    print(f"Frage: {QUESTION['content']}")
    print("Name 'Müller' und Anliegen 'Rezept' stehen nur in der Zusammenfassung.\n")
    for name, head in variants.items():
        messages = head + RECENT + [QUESTION]
        hits_name = 0
        hits_topic = 0
        print(f"== {name}")
        for run in range(1, RUNS + 1):
            answer = ask(base_url, llm, messages)
            name_ok = "müller" in answer.lower()
            topic_ok = "rezept" in answer.lower()
            hits_name += name_ok
            hits_topic += topic_ok
            print(f"  {run}. Name {'ja ' if name_ok else 'NEIN'} | Rezept {'ja ' if topic_ok else 'NEIN'} | {answer[:150]!r}")
        print(f"  -> Name erkannt {hits_name}/{RUNS}, Anliegen erkannt {hits_topic}/{RUNS}\n")


if __name__ == "__main__":
    main()

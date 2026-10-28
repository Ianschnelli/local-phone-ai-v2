"""Prüft, wie Zahlen vor Piper umgeschrieben werden (B-31).

Die Sätze stammen aus dem Live-Test vom 2026-10-25 (logs/app.log) und aus
typischen Öffnungszeiten-Antworten. Aufruf:

    .venv/Scripts/python eval/tts_numbers_check.py
"""

import asyncio
import sys

from local_phone_ai_v2.services.piper_tts import GermanPiperTTSService
from local_phone_ai_v2.services.tts_normalizer import normalize_tts_text

CASES = [
    # (Eingabe vom LLM, erwartete Ausgabe an die Stimme)
    (
        "Die Hausarztpraxis am Stadtpark befindet sich in der Beispielstraße 12, 74072 Heilbronn.",
        "Die Hausarztpraxis am Stadtpark befindet sich in der Beispielstraße 12, sieben vier null sieben zwei Heilbronn.",
    ),
    (
        "Unsere Telefonnummer lautet +49 7131 123456.",
        "Unsere Telefonnummer lautet plus vier neun sieben eins drei eins eins zwei drei vier fünf sechs.",
    ),
    (
        "Unsere Telefonnummer lautet 07131 123456.",
        "Unsere Telefonnummer lautet null sieben eins drei eins eins zwei drei vier fünf sechs.",
    ),
    # Regression: Satzende-Zahl darf keine Ordinalzahl werden.
    ("Die Sprechstunde ist von 8 bis 12.", "Die Sprechstunde ist von 8 bis 12."),
    ("Am Freitag bis 13 Uhr.", "Am Freitag bis 13 Uhr."),
    ("Montag bis Freitag von 8:00 bis 12:00 Uhr.", "Montag bis Freitag von acht Uhr bis zwölf Uhr."),
    # Ordinalzahlen mit Artikel bleiben erhalten.
    ("Die Praxis ist im 2. Stock.", "Die Praxis ist im zweiten Stock."),
    ("Am 3. Oktober ist geschlossen.", "Am dritten Oktober ist geschlossen."),
]


def check_texts() -> int:
    failures = 0
    for text, expected in CASES:
        result = normalize_tts_text(text)
        ok = result == expected
        failures += not ok
        print(("OK   " if ok else "FAIL ") + repr(result))
        if not ok:
            print("     erwartet: " + repr(expected))
    return failures


async def check_piper_uses_normalizer() -> int:
    """Stellt sicher, dass GermanPiperTTSService den Text normalisiert weitergibt."""
    seen: list[str] = []

    async def fake_run_tts(self, text, context_id):
        seen.append(text)
        if False:
            yield None

    original = GermanPiperTTSService.__mro__[1].run_tts
    GermanPiperTTSService.__mro__[1].run_tts = fake_run_tts
    try:
        service = GermanPiperTTSService.__new__(GermanPiperTTSService)
        async for _ in service.run_tts("Rufen Sie +49 7131 123456 an.", "ctx"):
            pass
    finally:
        GermanPiperTTSService.__mro__[1].run_tts = original

    ok = seen == ["Rufen Sie plus vier neun sieben eins drei eins eins zwei drei vier fünf sechs an."]
    print(("OK   " if ok else "FAIL ") + f"Piper erhält: {seen!r}")
    return 0 if ok else 1


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    failures = check_texts()
    failures += asyncio.run(check_piper_uses_normalizer())
    print(f"\n{len(CASES) + 1 - failures}/{len(CASES) + 1} bestanden")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

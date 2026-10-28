"""Ende-zu-Ende-Test: Öffnungszeiten, Adresse, Telefon und Rückfragen (Text, ohne Audio).

Jede Nutzeräußerung läuft durch den echten Wissens-Prozessor der App
(pipecat ``run_test``). Danach wird Qwen in LM Studio mit genau dem
Systemprompt der App gefragt. Der Gesprächsverlauf wird wie in der App
aufgebaut: Begrüßung, saubere Nutzertexte, echte Assistentenantworten.

Bewertung je Antwort:
- erwartete Angabe vorhanden (expect_any) und nichts Verbotenes (forbid)
- jede genannte Uhrzeit muss in facts.yaml vorkommen, sonst "erfundene Uhrzeit"
- jede Aussage "<Tag(e)> X bis Y" muss für alle genannten Tage stimmen, bei
  "Montag bis Freitag" müssen Ausnahmen genannt sein, sonst "falsch
  zusammengefasst" (B-27)
- Fälle mit "manual" werden nicht automatisch bewertet

Aufruf aus dem Projektverzeichnis (LM Studio muss laufen):
    .venv\\Scripts\\python.exe eval\\facts_e2e.py <bezeichnung>
Gespeicherte Antworten neu bewerten (ohne LLM):
    .venv\\Scripts\\python.exe eval\\facts_e2e.py --rescore <datei.json>
"""
import asyncio
import json
import os
import re
import statistics
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import yaml

os.environ.setdefault(
    "FASTEMBED_CACHE_PATH",
    str(Path("models/fastembed").resolve()),
)

FACTS_PATH = Path("customers/test_customer/facts.yaml")
RUNS = 3

# turns: Nutzeräußerungen in Reihenfolge, geprüft wird nur die Antwort auf die letzte.
# expect_any: mindestens einer dieser Texte muss in der Antwort vorkommen.
# forbid: keiner dieser Texte darf vorkommen.
# no_times: die Antwort darf keine Uhrzeit nennen.
# expect_digits: diese Ziffernfolge muss vorkommen (Leer- und Trennzeichen egal).
# manual: nicht automatisch bewertbar, Antwort von Hand prüfen.
CASES = [
    {
        "name": "Allgemein: Wann haben Sie geöffnet?",
        "turns": ["Wann haben Sie geöffnet?"],
    },
    {
        "name": "Terminwunsch (B-27/B-33)",
        "turns": ["Ich hätte gern einen Termin nächste Woche."],
        # Keine Buchung behaupten oder anbieten, keine ungefragten Zeiten,
        # aber an Anmeldung/Telefon verweisen (Entscheidung User 2026-10-26).
        "no_times": True,
        "forbid": [
            "habe ich gebucht", "ist gebucht", "eingetragen", "reserviert",
            "unterstützen, einen termin", "welchen tag", "welche uhrzeit",
        ],
        "expect_any": ["praxisteam", "anmeldung", "anmelden", "anrufen", "melden", "telefon", "wenden"],
    },
    {
        "name": "Samstag: Haben Sie samstags auf?",
        "turns": ["Haben Sie samstags auf?"],
        "expect_any": [
            "geschlossen",
            "nicht geöffnet",
            "nicht offen",
            "keine öffnungszeiten",
            "keine sprechzeiten",
            "keine sprechstunde",
        ],
    },
    {
        "name": "Blutabnahme: Wann kann ich zur Blutabnahme kommen?",
        "turns": ["Wann kann ich zur Blutabnahme kommen?"],
        "expect_any": ["halb acht", "7:30", "halb elf", "10:30"],
    },
    {
        "name": "Sprechstunde Freitag (direkt)",
        "turns": ["Haben Sie am Freitag Sprechstunde?"],
        "expect_any": ["zwölf", "12"],
        "forbid": ["keine sprechstunde", "nicht geöffnet"],
    },
    {
        "name": "Sonntag (nicht hinterlegt)",
        "turns": ["Ist die Praxis am Sonntag geöffnet?"],
        "manual": "darf keine Sonntagszeiten erfinden",
    },
    {
        "name": "Adresse: Wo finde ich Sie?",
        "turns": ["Wo finde ich Sie?"],
        "expect_any": ["beispielstraße"],
    },
    {
        "name": "Adresse: Wie komme ich zu Ihnen?",
        "turns": ["Wie komme ich zu Ihnen?"],
        "expect_any": ["beispielstraße"],
    },
    {
        "name": "Telefonnummer",
        "turns": ["Wie ist Ihre Telefonnummer?"],
        # Piper liest Ziffer für Ziffer (E-010), die Gruppierung ist egal.
        "expect_digits": "497131123456",
    },
    {
        "name": "Versichertennummer (nicht die Praxisnummer)",
        "turns": ["Was ist meine Versichertennummer?"],
        "forbid": ["sieben eins drei eins", "7131"],
    },
    {
        "name": "Rückfrage: Sprechstunde Montag -> Und freitags?",
        "turns": ["Wann ist die Sprechstunde am Montag?", "Und freitags?"],
        "expect_any": ["zwölf", "12"],
    },
    {
        "name": "Rückfrage: Anmeldung Mittwoch -> Und donnerstags?",
        "turns": ["Wann hat die Anmeldung am Mittwoch offen?", "Und donnerstags?"],
        "expect_any": ["achtzehn", "18"],
    },
    {
        "name": "Kontrolle Dokumentwissen: Hausbesuche",
        "turns": ["Machen Sie Hausbesuche?"],
        "expect_any": ["hausbesuch"],
    },
]

# ---------------------------------------------------------------- Uhrzeiten

NUMBER_WORDS = {
    "null": 0, "ein": 1, "eins": 1, "zwei": 2, "drei": 3, "vier": 4, "fünf": 5,
    "sechs": 6, "sieben": 7, "acht": 8, "neun": 9, "zehn": 10, "elf": 11,
    "zwölf": 12, "dreizehn": 13, "vierzehn": 14, "fünfzehn": 15,
    "sechzehn": 16, "siebzehn": 17, "achtzehn": 18, "neunzehn": 19,
    "zwanzig": 20, "einundzwanzig": 21, "zweiundzwanzig": 22,
    "dreiundzwanzig": 23, "vierundzwanzig": 24,
}
MINUTE_WORDS = {"dreißig": 30, "fünfzehn": 15, "fünfundvierzig": 45}
_WORD = "(" + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True)) + ")"
_WORD_NC = "(?:" + _WORD[1:]  # gleiche Liste, nicht einfangend (für Lookaheads)


def allowed_times() -> set[tuple[int, int]]:
    data = yaml.safe_load(FACTS_PATH.read_text(encoding="utf-8"))
    times = set()
    for department in (data.get("departments") or {}).values():
        if not isinstance(department, dict):
            continue
        for hours in (department.get("opening_hours") or {}).values():
            if not isinstance(hours, dict):
                continue
            for value in (hours.get("open"), hours.get("close")):
                if value:
                    hour, minute = value.split(":")
                    times.add((int(hour), int(minute)))
    return times


def extract_times(text: str) -> list[tuple[int, int]]:
    """Grobe Erkennung genannter Uhrzeiten (Ziffern und deutsche Zahlwörter)."""
    text = text.lower()
    found = []
    for h, m in re.findall(r"(?<![\d+])(\d{1,2})[:.](\d{2})(?!\d)", text):
        found.append((int(h), int(m)))
    for h in re.findall(r"(?<![\d:.+])(\d{1,2})\s*uhr", text):
        found.append((int(h), 0))
    for w in re.findall(rf"\bhalb {_WORD}\b", text):
        found.append((NUMBER_WORDS[w] - 1, 30))
    # "(?<!halb )": in "halb elf Uhr" ist "elf" keine eigene volle Stunde
    for w, minute in re.findall(rf"(?<!halb )\b{_WORD} uhr(?: (dreißig|fünfzehn|fünfundvierzig))?\b", text):
        found.append((NUMBER_WORDS[w], MINUTE_WORDS.get(minute, 0)))
    for w in re.findall(rf"(?<!halb )\b{_WORD}(?= bis (?:halb )?{_WORD_NC}\b)", text):
        found.append((NUMBER_WORDS[w], 0))
    return found


ALLOWED = allowed_times()

# ------------------------------------------- falsch zusammengefasste Tage (B-27)

WORKDAYS = {
    "monday": "montag", "tuesday": "dienstag", "wednesday": "mittwoch",
    "thursday": "donnerstag", "friday": "freitag",
}
_TIME_TOKEN = rf"(?:halb {_WORD_NC}|{_WORD_NC}(?: uhr(?: dreißig)?)?|\d{{1,2}}(?:[:.]\d{{2}})?(?: uhr)?)"


def workday_hours() -> dict[str, dict[str, tuple[int, int, int, int] | None]]:
    """Je Bereich: Wochentag -> (h_auf, m_auf, h_zu, m_zu) oder None."""
    data = yaml.safe_load(FACTS_PATH.read_text(encoding="utf-8"))
    result = {}
    for key, department in (data.get("departments") or {}).items():
        if not isinstance(department, dict):
            continue
        hours = department.get("opening_hours") or {}
        days = {}
        for day in WORKDAYS:
            value = hours.get(day)
            if isinstance(value, dict) and value.get("open") and value.get("close"):
                (ho, mo), (hc, mc) = (map(int, value[k].split(":")) for k in ("open", "close"))
                days[day] = (ho, mo, hc, mc)
            else:
                days[day] = None
        result[key] = days
    return result


WORKDAY_HOURS = workday_hours()


def parse_time(token: str) -> tuple[int, int] | None:
    token = token.replace(" uhr", "").strip()
    if token.startswith("halb "):
        return NUMBER_WORDS[token[5:]] - 1, 30
    match = re.fullmatch(r"(\d{1,2})(?:[:.](\d{2}))?", token)
    if match:
        return int(match.group(1)), int(match.group(2) or 0)
    word, _, minute = token.partition(" ")
    if word in NUMBER_WORDS:
        return NUMBER_WORDS[word], 30 if minute == "dreißig" else 0
    return None


_DAY = r"(?:montag|dienstag|mittwoch|donnerstag|freitag)s?"
_DAY_ORDER = list(WORKDAYS)
_NAME_TO_DAY = {name: day for day, name in WORKDAYS.items()}


def _days_in(expr: str) -> tuple[list[str], bool]:
    """Wochentage eines Ausdrucks wie "montag bis freitag" oder "montag, dienstag und freitag"."""
    names = [n.rstrip("s") for n in re.findall(_DAY, expr)]
    days = [_NAME_TO_DAY[n] for n in names]
    if " bis " in expr and len(days) == 2:
        i, j = _DAY_ORDER.index(days[0]), _DAY_ORDER.index(days[1])
        return _DAY_ORDER[i:j + 1], True
    return days, False


def wrong_week_range(text: str) -> str | None:
    """Prüft jede Aussage "<Tag(e)> ... X bis Y" gegen facts.yaml.

    Richtig, wenn ein Bereich an allen genannten Tagen genau diese Zeit hat.
    Bei einer Spanne ("montags bis freitags") reicht es auch, wenn alle
    abweichenden Tage dieses Bereichs an anderer Stelle der Antwort genannt
    werden ("außer Mittwoch ..."). Sonst falsch zusammengefasst.
    """
    days_expr = rf"{_DAY}(?: bis {_DAY}|(?:(?:, | und ){_DAY})*)"
    gap = rf"(?:(?!{_DAY})[^.!?])*?"
    pattern = rf"({days_expr}){gap}({_TIME_TOKEN}) (?:bis|und) ({_TIME_TOKEN})(?![\w])"
    for match in re.finditer(pattern, text):
        start, end = parse_time(match.group(2)), parse_time(match.group(3))
        if not start or not end:
            continue
        days, is_range = _days_in(match.group(1))
        claimed = (*start, *end)
        shown = f"{match.group(1)} {start[0]}:{start[1]:02d}–{end[0]}:{end[1]:02d}"
        if any(all(hours[d] == claimed for d in days) for hours in WORKDAY_HOURS.values()):
            continue
        if is_range:
            # Ausnahmen müssen außerhalb dieser Aussage genannt sein.
            rest = text[:match.start(1)] + text[match.end(1):]
            for hours in WORKDAY_HOURS.values():
                deviating = [WORKDAYS[d] for d in days if hours[d] != claimed]
                if len(deviating) < len(days) and all(name in rest for name in deviating):
                    break
            else:
                return f"{shown}, Ausnahmen fehlen"
            continue
        return f"{shown} stimmt nicht"

    # Nur Schlusszeit: "mittwochs bis zwölf Uhr", "donnerstag bis 18 Uhr"
    for match in re.finditer(rf"({days_expr}) bis ({_TIME_TOKEN})(?![\w])", text):
        end = parse_time(match.group(2))
        if not end:
            continue
        days, _ = _days_in(match.group(1))
        if not any(
            all(hours[d] is not None and hours[d][2:] == end for d in days)
            for hours in WORKDAY_HOURS.values()
        ):
            return f"{match.group(1)} bis {end[0]}:{end[1]:02d} stimmt nicht"
    return None


def judge(case: dict, answer: str) -> tuple[bool | None, str]:
    if case.get("manual"):
        return None, "von Hand prüfen"
    text = answer.lower()
    bad_times = sorted({t for t in extract_times(text) if t not in ALLOWED})
    if bad_times:
        shown = ", ".join(f"{h}:{m:02d}" for h, m in bad_times)
        return False, f"erfundene Uhrzeit ({shown})"
    wrong_range = wrong_week_range(text)
    if wrong_range:
        return False, f"falsch zusammengefasst ({wrong_range})"
    if any(word in text for word in case.get("forbid", [])):
        return False, "verbotene Angabe"
    if case.get("no_times") and extract_times(text):
        return False, "ungefragt Uhrzeiten genannt"
    digits = case.get("expect_digits")
    if digits and digits not in re.sub(r"\D", "", text):
        return False, "Ziffernfolge fehlt"
    expect = case.get("expect_any")
    if expect and not any(word in text for word in expect):
        return False, "erwartete Angabe fehlt"
    return True, "ok"


# ---------------------------------------------------------------- Lauf

async def augment(config, resources, messages: list[dict]) -> list[dict]:
    """Schickt den Kontext durch den echten Wissens-Prozessor und gibt zurück, was beim LLM ankommt."""
    from pipecat.frames.frames import LLMContextFrame
    from pipecat.processors.aggregators.llm_context import LLMContext
    from pipecat.tests.utils import run_test

    from local_phone_ai_v2.pipeline.build import create_knowledge_processor

    processor = create_knowledge_processor(config, resources)
    context = LLMContext(messages=[dict(m) for m in messages])
    down, _up = await run_test(processor, frames_to_send=[LLMContextFrame(context=context)])
    return processor_output(down, messages)


def processor_output(down: list, messages: list[dict]) -> list[dict]:
    """Was beim LLM ankommt. Eine feste Ansage (Notfall, TTSSpeakFrame) wird
    als Nachricht mit "_fixed" zurückgegeben; ask_llm nimmt sie dann direkt."""
    from pipecat.frames.frames import LLMContextFrame, TTSSpeakFrame

    spoken = [f for f in down if isinstance(f, TTSSpeakFrame)]
    if spoken:
        return [*messages, {"role": "assistant", "content": spoken[-1].text, "_fixed": True}]
    frames = [f for f in down if isinstance(f, LLMContextFrame)]
    if not frames:
        return [dict(m) for m in messages]
    return frames[-1].context.get_messages()


def ask_llm(llm: dict, system_instruction: str, messages: list[dict]) -> tuple[str, float]:
    if messages and messages[-1].get("_fixed"):
        return messages[-1]["content"], 0.0
    payload = {
        "model": llm["model"],
        "messages": [{"role": "system", "content": system_instruction}, *messages],
        "temperature": llm["temperature"],
        "top_p": llm["top_p"],
        "max_tokens": llm["max_tokens"],
        "stream": True,
    }
    request = urllib.request.Request(
        f"{llm['base_url'].rstrip('/')}/chat/completions",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first = None
    text = ""
    with urllib.request.urlopen(request, timeout=180) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            chunk = json.loads(line[5:])
            if "choices" not in chunk:
                raise RuntimeError(f"LM Studio meldet: {line[:300]}")
            delta = chunk["choices"][0]["delta"].get("content") or ""
            if delta and first is None:
                first = time.perf_counter() - start
            text += delta
    return text.strip(), (first if first is not None else float("nan"))


def describe_knowledge(sent_last: dict, clean_text: str) -> str:
    content = sent_last.get("content")
    if content == clean_text:
        return "kein Wissensblock"
    parts = []
    if "EXAKTE FAKTEN:" in content:
        facts = content.split("EXAKTE FAKTEN:")[1].split("DOKUMENTWISSEN:")[0]
        parts.append(f"Fakten: {sum(1 for line in facts.splitlines() if line.startswith('- '))} Zeile(n)")
    if "DOKUMENTWISSEN:" in content:
        parts.append("Dokumentwissen: ja")
    return ", ".join(parts) or "Wissensblock (unbekanntes Format)"


def print_report(label: str, results: list[dict], system_chars: int | None) -> tuple[int, int]:
    head = f"Lauf: {label}"
    if system_chars is not None:
        head += f" | Systemprompt: {system_chars} Zeichen"
    print(head + f" | {RUNS} Läufe je Fall\n")
    passed = total = 0
    for result in results:
        oks = [run["ok"] for run in result["runs"]]
        if result.get("manual"):
            verdict = f"von Hand prüfen ({result['manual']})"
        else:
            verdict = f"{sum(1 for ok in oks if ok)}/{len(oks)} richtig"
            passed += sum(1 for ok in oks if ok)
            total += len(oks)
        print(f"== {result['name']}  ->  {verdict}")
        print(f"   Wissen beim letzten Turn: {result['runs'][0]['knowledge']}")
        for i, run in enumerate(result["runs"], start=1):
            mark = {True: "ok  ", False: "FEHL", None: "?   "}[run["ok"]]
            reason = "" if run["ok"] in (True, None) else f" [{run['reason']}]"
            print(f"   {i}. {mark}{reason} {run['answer'][:170]!r}")
        print()
    print(f"GESAMT (automatisch bewertet): {passed}/{total} richtig")
    return passed, total


def rescore(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = {case["name"]: case for case in CASES}
    results = []
    for result in data["results"]:
        case = cases[result["name"]]
        runs = []
        for run in result["runs"]:
            ok, reason = judge(case, run["answer"])
            runs.append({**run, "ok": ok, "reason": reason})
        results.append({**case, "runs": runs})
    passed, total = print_report(f"{data['label']} (neu bewertet)", results, data.get("system_instruction_chars"))
    out = path.with_name(path.stem + "_neu-bewertet.json")
    out.write_text(
        json.dumps({**data, "passed": passed, "total": total, "results": results}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"Neu bewertet gespeichert: {out}")


async def main(label: str) -> None:
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    from local_phone_ai_v2.config.loader import load_config
    from local_phone_ai_v2.conversation.customer_loader import load_customer_config
    from local_phone_ai_v2.pipeline.build import create_system_instruction
    from local_phone_ai_v2.runtime.resources import RuntimeResources

    config = load_config("config.yaml")
    llm = config["llm"]
    customer = load_customer_config(Path("customers"), "test_customer")

    resources = RuntimeResources(config)
    resources.warmup()
    system_instruction = create_system_instruction(customer["instructions"], resources)

    results = []
    for case in CASES:
        runs = []
        for _ in range(RUNS):
            messages = [{"role": "assistant", "content": customer["greeting"]}]
            answer = ""
            knowledge = ""
            ttft = float("nan")
            for user_text in case["turns"]:
                messages.append({"role": "user", "content": user_text})
                sent = await augment(config, resources, messages)
                knowledge = describe_knowledge(sent[-1], user_text)
                answer, ttft = ask_llm(llm, system_instruction, sent)
                messages.append({"role": "assistant", "content": answer})
            ok, reason = judge(case, answer)
            runs.append({"answer": answer, "knowledge": knowledge, "ttft": ttft, "ok": ok, "reason": reason})
        results.append({**case, "runs": runs})

    passed, total = print_report(label, results, len(system_instruction))
    ttfts = [run["ttft"] for r in results for run in r["runs"]]
    print(f"Erstes Token LLM: Median {statistics.median(ttfts):.2f}s, Max {max(ttfts):.2f}s")

    today = datetime.now().astimezone().date().isoformat()
    out = Path(f"docs/rag/daten/{today}_facts_e2e_{label}.json")
    out.write_text(
        json.dumps(
            {
                "label": label,
                "system_instruction": system_instruction,
                "system_instruction_chars": len(system_instruction),
                "passed": passed,
                "total": total,
                "results": results,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"Rohdaten: {out}")
    resources.close()


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--rescore":
        rescore(Path(sys.argv[2]))
    else:
        asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "ohne-bezeichnung"))

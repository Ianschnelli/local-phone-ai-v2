"""Gesprächskontext über das ganze Telefonat (E-019).

Mehrschritt-Gespräche mit der großen Testpraxis: Eine Angabe aus dem ersten
Satz des Anrufers wird erst im letzten Schritt gebraucht. Alle Zwischen-
antworten erzeugt Qwen selbst; der Wissens-Prozessor lebt wie in der App
über das ganze Gespräch. Bewertet wird die letzte Antwort; zusätzlich darf die
Antwort auf "Okay, danke." kein Wissen zum vorigen Thema enthalten (Review E-023).

Aufruf (LM Studio muss laufen):
    .venv\\Scripts\\python.exe eval\\context_e2e.py <bezeichnung> [--runs 3] [--set rag.reranker.min_score=0.002]
"""
import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import facts_e2e
from rag_e2e import apply_overrides, augment_with

FILLER = [
    "Wie lange dauert es, bis Laborergebnisse da sind?",
    "Wo kann man bei Ihnen parken?",
    "Gibt es bei Ihnen einen Aufzug?",
    "Okay, danke.",
]

CONVERSATIONS = [
    {
        "name": "Bettlägeriger Vater → Blutabnahme zu Hause",
        "first": "Guten Tag, mein Vater ist seit einem Schlaganfall bettlägerig.",
        "last": "Kann dann eigentlich jemand zu ihm zum Blutabnehmen kommen?",
        "expect_any": ["möglich", "nach hause", "zu hause", "hausbesuch", "zu ihnen", "zu ihm"],
        "forbid": ["nicht möglich", "keine information"],
    },
    {
        "name": "Privat versichert → Rechnung",
        "first": "Hallo, ich bin übrigens privat versichert.",
        "last": "Wie bekomme ich denn bei Ihnen meine Rechnung?",
        "expect_any": ["post", "goä", "gebührenordnung", "rechnung"],
        "forbid": ["keine information"],
    },
    {
        "name": "Marcumar → Wert kontrollieren",
        "first": "Ich nehme seit einem Jahr Marcumar.",
        "last": "Wo kann ich dann meinen Wert kontrollieren lassen?",
        "expect_any": ["inr", "fingerkuppe", "in der praxis", "bei uns"],
        "forbid": ["keine information"],
    },
    {
        "name": "Name aus dem ersten Satz",
        "first": "Guten Tag, mein Name ist Peter Schmidt.",
        "last": "Welchen Namen hatte ich Ihnen am Anfang gesagt?",
        "expect_any": ["schmidt"],
        "forbid": [],
    },
    {
        "name": "67 Jahre → Grippeimpfung zahlt Kasse",
        "first": "Ich bin 67 Jahre alt.",
        "last": "Muss ich die Grippeimpfung dann selbst bezahlen?",
        "expect_any": ["kasse", "krankenkasse", "übernimmt", "nicht selbst", "kostenlos"],
        "forbid": ["keine information"],
    },
    {
        "name": "15 Jahre → kein Patient",
        "first": "Hallo, ich bin 15 Jahre alt.",
        "last": "Kann ich bei Ihnen Patient werden?",
        "expect_any": ["16", "sechzehn", "kinderarzt", "kinderärztlich"],
        "forbid": [],
    },
    {
        "name": "Langzeit-EKG am Anfang → duschen am Ende",
        "first": "Ich bekomme morgen bei Ihnen ein Langzeit-EKG.",
        "last": "Darf ich damit eigentlich duschen?",
        "expect_any": ["nicht"],
        "forbid": ["ja, sie dürfen", "keine information", "wasserdicht"],
    },
    {
        "name": "Belastungs-EKG am Anfang → Kleidung am Ende",
        "first": "Ich habe nächste Woche ein Belastungs-EKG bei Ihnen.",
        "last": "Was soll ich dafür eigentlich anziehen?",
        "expect_any": ["sportschuhe", "bequem", "kleidung"],
        "forbid": ["keine information"],
    },
]


def judge(conv: dict, answer: str) -> tuple[bool, str]:
    text = answer.lower()
    if any(word in text for word in conv["forbid"]):
        return False, "verbotene Angabe"
    if not any(word in text for word in conv["expect_any"]):
        return False, "erwartete Angabe fehlt"
    return True, "ok"


async def main(args: argparse.Namespace) -> None:
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    from local_phone_ai_v2.config.loader import load_config
    from local_phone_ai_v2.conversation.customer_loader import load_customer_config
    from local_phone_ai_v2.pipeline.build import (
        create_knowledge_processor,
        create_system_instruction,
    )
    from local_phone_ai_v2.runtime.resources import RuntimeResources

    config = load_config("config.yaml")
    config["rag"]["qdrant_path"] = args.qdrant
    config["customer"]["facts_path"] = str(Path("customers") / args.customer / "facts.yaml")
    apply_overrides(config, args.set)
    customer = load_customer_config(Path("customers"), args.customer)
    resources = RuntimeResources(config)
    resources.warmup()
    system_instruction = create_system_instruction(customer["instructions"], resources)

    results = []
    for conv in CONVERSATIONS:
        runs = []
        for _ in range(args.runs):
            processor = create_knowledge_processor(config, resources)
            messages = [{"role": "assistant", "content": customer["greeting"]}]
            transcript = []
            for user_text in [conv["first"], *FILLER, conv["last"]]:
                messages.append({"role": "user", "content": user_text})
                sent = await augment_with(processor, messages)
                answer, _ttft = facts_e2e.ask_llm(config["llm"], system_instruction, sent)
                messages.append({"role": "assistant", "content": answer})
                transcript.append({"user": user_text, "assistant": answer})
            ok, reason = judge(conv, transcript[-1]["assistant"])
            thanks = next(t["assistant"] for t in transcript if t["user"] == "Okay, danke.")
            if ok and any(word in thanks.lower() for word in ("aufzug", "obergeschoss", "barrierefrei")):
                ok, reason = False, "Antwort auf 'Okay, danke.' enthält altes Wissen"
            runs.append({"ok": ok, "reason": reason, "transcript": transcript})
        results.append({**conv, "runs": runs})
        passed = sum(1 for r in runs if r["ok"])
        print(f"{passed}/{len(runs)}  {conv['name']}")
        for run in runs:
            if not run["ok"]:
                print(f"      [{run['reason']}] {run['transcript'][-1]['assistant'][:200]!r}")

    total = sum(1 for r in results for run in r["runs"] if run["ok"])
    count = sum(len(r["runs"]) for r in results)
    print(f"GESAMT {total}/{count}")
    today = datetime.now().astimezone().date().isoformat()
    out = Path(f"docs/rag/daten/{today}_context_e2e_{args.label}.json")
    out.write_text(json.dumps({"label": args.label, "overrides": args.set, "passed": total, "total": count,
                               "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Rohdaten: {out}")
    resources.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("label", nargs="?", default="ohne-bezeichnung")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--customer", default="test_praxis_gross")
    parser.add_argument("--qdrant", default="data/qdrant_gross_v2")
    parser.add_argument("--set", action="append", default=[])
    asyncio.run(main(parser.parse_args()))

"""Langes Telefonat mit der Zusammenfassungs-Logik der App (E-020).

Der Anrufer nennt im ersten Satz Name, Alter und Versicherung, stellt dann
viele Fragen (aus dem Testset der großen Praxis) und fragt am Ende nach
Name und Alter. Nach jeder Antwort läuft dieselbe Prüfung wie in
``CallSession`` (``should_compact`` / ``prepare_compaction``), inkl. echter
Zusammenfassung durch Qwen. Protokolliert Token, Zusammenfassungen und ob
der Kontext überläuft.

Aufruf (LM Studio muss laufen):
    .venv\\Scripts\\python.exe eval\\longcall_e2e.py <bezeichnung> [--turns 30] [--runs 2] [--set ...]
"""
import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))

import facts_e2e
from rag_e2e import apply_overrides, augment_with

FIRST = "Guten Tag, mein Name ist Peter Schmidt, ich bin 67 Jahre alt und privat versichert."
FINAL = [
    ("Wie war nochmal mein Name, den ich am Anfang gesagt habe?", ["schmidt"]),
    ("Und wie alt hatte ich gesagt, dass ich bin?", ["67", "siebenundsechzig"]),
]


def filler_questions(count: int) -> list[str]:
    items = yaml.safe_load(Path("eval/rag_testset_gross.yaml").read_text(encoding="utf-8"))["questions"]
    picked = [i["q"] for i in items if i["category"] in ("soll", "umgangssprachlich") and not i.get("safety")]
    return picked[:count]


async def run_call(config, resources, system_instruction, greeting, turns: int) -> dict:
    from local_phone_ai_v2.conversation.context_manager import (
        ConversationContextManager,
    )
    from local_phone_ai_v2.pipeline.build import create_knowledge_processor

    llm = config["llm"]
    conversation = config["conversation"]
    manager = ConversationContextManager(
        tokenizer_model=llm["tokenizer_model"],
        context_window=llm["context_window"],
        response_reserve_tokens=llm["response_reserve_tokens"],
        compact_at_ratio=conversation["compact_at_ratio"],
        recent_turn_pairs=conversation["recent_turn_pairs"],
        max_compactions=conversation["max_compactions"],
        summarizer=resources.summarizer,
        system_instruction=system_instruction,
        rag_reserve_tokens=conversation["rag_reserve_tokens"],
    )
    processor = create_knowledge_processor(config, resources)
    messages = [{"role": "assistant", "content": greeting}]
    events = []
    user_turns = [FIRST, *filler_questions(turns - 1 - len(FINAL)), *(q for q, _ in FINAL)]

    answers = []
    for index, user_text in enumerate(user_turns, start=1):
        messages.append({"role": "user", "content": user_text})
        sent = await augment_with(processor, messages)
        total_tokens = manager.count_tokens([{"role": "system", "content": system_instruction}, *sent])
        try:
            answer, _ = facts_e2e.ask_llm(llm, system_instruction, sent)
        except Exception as exc:  # noqa: BLE001
            events.append({"turn": index, "event": "LLM-Fehler", "detail": str(exc)[:200], "tokens": total_tokens})
            answer = ""
        messages.append({"role": "assistant", "content": answer})
        answers.append({"turn": index, "user": user_text, "assistant": answer, "tokens": total_tokens})

        if manager.should_compact(messages):
            if manager.can_compact():
                summary, recent = manager.prepare_compaction(messages)
                manager.commit_compaction(summary)
                summary_message = manager.build_summary_message()
                messages = ([summary_message] if summary_message else []) + recent
                events.append({"turn": index, "event": "zusammengefasst", "summary": summary.model_dump()})
            elif not any(e["event"] == "Grenze erreicht" for e in events):
                events.append({"turn": index, "event": "Grenze erreicht", "tokens": total_tokens})

    checks = []
    for (question, expect), answer in zip(FINAL, answers[-len(FINAL):]):
        ok = any(word in answer["assistant"].lower() for word in expect)
        checks.append({"question": question, "ok": ok, "answer": answer["assistant"]})
    return {"checks": checks, "events": events, "answers": answers}


async def main(args: argparse.Namespace) -> None:
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    from local_phone_ai_v2.config.loader import load_config
    from local_phone_ai_v2.conversation.customer_loader import load_customer_config
    from local_phone_ai_v2.pipeline.build import create_system_instruction
    from local_phone_ai_v2.runtime.resources import RuntimeResources

    config = load_config("config.yaml")
    config["rag"]["qdrant_path"] = args.qdrant
    config["customer"]["facts_path"] = str(Path("customers") / args.customer / "facts.yaml")
    apply_overrides(config, args.set)
    customer = load_customer_config(Path("customers"), args.customer)
    resources = RuntimeResources(config)
    resources.warmup()
    system_instruction = create_system_instruction(customer["instructions"], resources)

    runs = []
    for run_index in range(args.runs):
        result = await run_call(config, resources, system_instruction, customer["greeting"], args.turns)
        runs.append(result)
        print(f"Lauf {run_index + 1}: max. Token {max(a['tokens'] for a in result['answers'])}")
        for event in result["events"]:
            print(f"   Schritt {event['turn']}: {event['event']} {event.get('detail', '')}")
        for check in result["checks"]:
            mark = "ok  " if check["ok"] else "FEHL"
            print(f"   {mark} {check['question']} -> {check['answer'][:150]!r}")

    total = sum(1 for r in runs for c in r["checks"] if c["ok"])
    count = sum(len(r["checks"]) for r in runs)
    print(f"GESAMT {total}/{count}")
    today = datetime.now().astimezone().date().isoformat()
    out = Path(f"docs/rag/daten/{today}_longcall_{args.label}.json")
    out.write_text(json.dumps({"label": args.label, "turns": args.turns, "overrides": args.set,
                               "passed": total, "total": count, "runs": runs},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Rohdaten: {out}")
    resources.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("label", nargs="?", default="ohne-bezeichnung")
    parser.add_argument("--turns", type=int, default=30)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--customer", default="test_praxis_gross")
    parser.add_argument("--qdrant", default="data/qdrant_gross_v2")
    parser.add_argument("--set", action="append", default=[])
    asyncio.run(main(parser.parse_args()))

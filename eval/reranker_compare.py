"""Vergleich Reranker alt (mmarco-MiniLM) gegen bge-reranker-v2-m3 (E-013).

Nur Messung, ändert weder Index noch App. Holt für jede Testfrage die
Kandidaten aus dem echten Retriever (candidate_k aus der Config und 16) und
bewertet sie mit beiden Modellen. Ausgabe: Rang des Soll-Chunks, Gate-Sweep
und Latenz.

Aufruf aus dem Projektverzeichnis:
    .venv\\Scripts\\python.exe eval\\reranker_compare.py [cpu|cuda]
"""
import copy
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("FASTEMBED_CACHE_PATH", str(Path("models/fastembed").resolve()))

import yaml
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")

from sentence_transformers import CrossEncoder

from local_phone_ai_v2.config.loader import load_config
from local_phone_ai_v2.knowledge.query_builder import build_retrieval_query
from local_phone_ai_v2.knowledge.retriever import KnowledgeRetriever

TESTSET = Path("eval/rag_testsetALTIN.yaml")
NEW_MODEL = "BAAI/bge-reranker-v2-m3"


def chunk_key(payload: dict) -> tuple[str, str]:
    headings = (payload.get("docling") or {}).get("headings") or [""]
    return payload.get("source", ""), headings[-1]


def rank_of_gold(scores: list[float], gold_flags: list[bool]) -> int | None:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return next((pos + 1 for pos, i in enumerate(order) if gold_flags[i]), None)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("device", nargs="?", default="cpu")
    parser.add_argument("--testset", default=str(TESTSET))
    parser.add_argument("--qdrant", default=None)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    device = args.device
    config = load_config("config.yaml")
    if args.qdrant:
        config["rag"]["qdrant_path"] = args.qdrant
    questions = yaml.safe_load(Path(args.testset).read_text(encoding="utf-8"))["questions"]

    wide = copy.deepcopy(config)
    wide["rag"]["candidate_k"] = 16
    retriever = KnowledgeRetriever(wide)
    retriever.warmup()

    points, _ = retriever._client.scroll(
        collection_name=config["rag"]["collection"], limit=1000, with_payload=True, with_vectors=False
    )
    key_by_hash = {p.payload.get("content_hash"): chunk_key(p.payload) for p in points}

    old = CrossEncoder(config["rag"]["reranker"]["model"], device=device, local_files_only=True)
    new = CrossEncoder(NEW_MODEL, device=device, local_files_only=True)
    for model in (old, new):
        model.predict([("warmup", "warmup")] * 8)

    k = config["rag"]["candidate_k"]
    records = []
    t_old, t_new = [], []
    for item in questions:
        q = build_retrieval_query([*item.get("history", []), {"role": "user", "content": item["q"]}])
        gold = {tuple(g) for g in item.get("gold", [])}
        hits = retriever.search(q)  # bis zu 16 Kandidaten, Reihenfolge = Hybrid-RRF
        flags = [key_by_hash.get(h.content_hash) in gold for h in hits]

        top = hits[:k]
        t0 = time.perf_counter()
        old_raw = old.predict([(q, h.text) for h in top])
        old_emb = old.predict([(q, h.embedding_text) for h in top])
        t1 = time.perf_counter()
        new_emb = new.predict([(q, h.embedding_text) for h in top])
        t2 = time.perf_counter()
        t_old.append((t1 - t0) * 1000)
        t_new.append((t2 - t1) * 1000)

        records.append({
            "q": q,
            "category": item["category"],
            "set": item.get("set", "e001"),
            "safety": bool(item.get("safety")),
            "has_gold": bool(gold),
            "gold_in_8": any(flags[:k]),
            "gold_in_16": any(flags),
            "flags": flags[:k],
            "old": [float((r + e) / 2) for r, e in zip(old_raw, old_emb)],
            "new": [float(s) for s in new_emb],
        })

    pos = [r for r in records if r["has_gold"]]
    neg = [r for r in records if not r["has_gold"]]
    groups = {
        "soll": [r for r in pos if r["category"] == "soll"],
        "umgangssprachlich": [r for r in pos if r["category"] == "umgangssprachlich"],
        "rueckfrage": [r for r in pos if r["category"] == "rueckfrage"],
        "safety": [r for r in pos if r["safety"]],
    }

    print(f"Gerät: {device} | {len(records)} Fragen | Recall@{k}: "
          f"{sum(r['gold_in_8'] for r in pos)}/{len(pos)}  Recall@16: {sum(r['gold_in_16'] for r in pos)}/{len(pos)}")

    print("\n== Rang des Soll-Chunks (Rang 1 / Top 3) ==")
    for model in ("old", "new"):
        line = f"  {model:<4}"
        for name, rows in groups.items():
            ranks = [rank_of_gold(r[model], r["flags"]) for r in rows]
            r1 = sum(1 for x in ranks if x == 1)
            r3 = sum(1 for x in ranks if x and x <= 3)
            line += f" | {name} {r1}/{len(rows)} · {r3}/{len(rows)}"
        print(line)

    print("\n== Bester Score: Soll-Fragen (Soll-Chunk auf Rang 1) gegen Negativfragen ==")
    for model in ("old", "new"):
        good = sorted(max(r[model]) for r in pos if rank_of_gold(r[model], r["flags"]) == 1)
        bad = sorted(max(r[model]) for r in neg)
        print(f"  {model:<4} Soll min {good[0]:.3f} · Median {statistics.median(good):.3f} | "
              f"Negativ Median {statistics.median(bad):.3f} · max {bad[-1]:.3f}")

    print("\n== Gate-Sweep neues Modell (Top-1 >= Schwelle, dann bis zu 3 Chunks >= Schwelle) ==")
    print(f"  {'Schwelle':>8} {'soll':>7} {'umg.':>7} {'Safety':>7} {'Neg. leer':>10}")
    for threshold in (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5):
        def hit(rows, th=threshold):
            n = 0
            for r in rows:
                sel = [i for i in sorted(range(len(r["new"])), key=lambda i: r["new"][i], reverse=True)
                       if r["new"][i] >= th][:3]
                n += any(r["flags"][i] for i in sel)
            return n
        empty = sum(1 for r in neg if max(r["new"]) < threshold)
        print(f"  {threshold:>8} {hit(groups['soll']):>3}/{len(groups['soll']):<3} "
              f"{hit(groups['umgangssprachlich']):>3}/{len(groups['umgangssprachlich']):<3} "
              f"{hit(groups['safety']):>3}/{len(groups['safety']):<3} {empty:>5}/{len(neg):<4}")

    print("\n== Negativfragen: bester Score neu ==")
    for r in sorted(neg, key=lambda r: -max(r["new"])):
        print(f"  {max(r['new']):.3f}  {r['q']}")

    print(f"\n== Latenz Rerank ({k} Kandidaten, ms) ==")
    print(f"  alt (2 Durchläufe raw+emb): Median {statistics.median(t_old):.0f}  Max {max(t_old):.0f}")
    print(f"  neu (1 Durchlauf emb):      Median {statistics.median(t_new):.0f}  Max {max(t_new):.0f}")

    out = Path(f"docs/rag/daten/{datetime.now().astimezone().date().isoformat()}_reranker_compare_{device}{('_' + args.label) if args.label else ''}.json")
    out.write_text(json.dumps({"device": device, "records": records,
                               "latency_ms": {"old": t_old, "new": t_new}},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nRohdaten: {out}")
    retriever.close()


if __name__ == "__main__":
    main()

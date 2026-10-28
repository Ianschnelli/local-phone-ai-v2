"""RAG-Evaluation ohne LLM und ohne Audio.

Misst Retrieval -> Rerank -> Gate gegen eval/rag_testsetALTIN.yaml und gibt
Kennzahlen plus Latenz aus. Ändert weder Index noch Code.

Aufruf aus dem Projektverzeichnis:
    .venv\\Scripts\\python.exe eval\\rag_evalALTIN.py [ausgabe.json]

Optionen: --testset eval/rag_testset_gross.yaml --qdrant data/qdrant_gross
(große Testpraxis, E-014). Fragen mit "history" (Rückfragen) gehen wie in
der App durch build_retrieval_query, also mit Gesprächsverlauf.

Hinweis: Greift auf Reranker-Interna (_model) zu, um raw- und
embedding_text-Score getrennt zu sehen. Das ist nur für die Diagnose.
"""
import argparse
import json
import os
import statistics
import sys
import time
from datetime import date
from pathlib import Path

# fastembed legt Modelle sonst im OS-Temp-Ordner ab (siehe docs/BEFUNDEALTIN.md B-01).
os.environ.setdefault(
    "FASTEMBED_CACHE_PATH",
    str(Path("models/fastembed").resolve()),
)

import yaml
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")

from local_phone_ai_v2.config.loader import load_config
from local_phone_ai_v2.knowledge.query_builder import (
    build_retrieval_query,
)
from local_phone_ai_v2.knowledge.reranker import KnowledgeReranker
from local_phone_ai_v2.knowledge.retriever import KnowledgeRetriever

TESTSET = Path("eval/rag_testsetALTIN.yaml")
DEFAULT_OUT = Path(f"docs/rag/daten/{date.today().isoformat()}_rag_evalALTIN.json")

VARIANTS = {
    "avg(raw,emb) [aktuell]": lambda raw, emb: (raw + emb) / 2,
    "nur raw (Text)": lambda raw, emb: raw,
    "nur emb (Kontext)": lambda raw, emb: emb,
    "max(raw,emb)": lambda raw, emb: max(raw, emb),
}


def key(meta: dict) -> tuple[str, str]:
    headings = meta.get("headings") or [""]
    return (meta.get("source", ""), headings[-1])


# Gate-Varianten (E-013): (Name, Score-Funktion, min_score oder None = kein Gate, Fenster, k)
GATES = [
    ("aktuell: avg, min -2.3, Fenster 1.5, k3", VARIANTS["avg(raw,emb) [aktuell]"], -2.3, 1.5, 3),
    ("avg, min -3.0", VARIANTS["avg(raw,emb) [aktuell]"], -3.0, 1.5, 3),
    ("avg, min -4.0", VARIANTS["avg(raw,emb) [aktuell]"], -4.0, 1.5, 3),
    ("avg, min -5.0", VARIANTS["avg(raw,emb) [aktuell]"], -5.0, 1.5, 3),
    ("max, min -2.3", VARIANTS["max(raw,emb)"], -2.3, 1.5, 3),
    ("max, min -3.0", VARIANTS["max(raw,emb)"], -3.0, 1.5, 3),
    ("max, min -4.0", VARIANTS["max(raw,emb)"], -4.0, 1.5, 3),
    ("kein Gate, immer Top-2", VARIANTS["avg(raw,emb) [aktuell]"], None, None, 2),
    ("kein Gate, immer Top-3", VARIANTS["avg(raw,emb) [aktuell]"], None, None, 3),
]


def simulate_gate(candidates: list[dict], fn, min_score, window, k) -> list[dict]:
    ranked = sorted(candidates, key=lambda c: fn(c["raw"], c["emb"]), reverse=True)
    if not ranked:
        return []
    if min_score is None:
        return ranked[:k]
    top = fn(ranked[0]["raw"], ranked[0]["emb"])
    if top < min_score:
        return []
    floor = max(min_score, top - window)
    return [c for c in ranked if fn(c["raw"], c["emb"]) >= floor][:k]


def print_gate_comparison(records: list[dict], min_score, window, final_k) -> None:
    print("\n== Gate-Varianten (Soll-Chunk im Kontext / Negativ korrekt leer) ==")
    groups = {
        "E-001 alt": [r for r in records if r["set"] == "e001"],
        "umgangsspr. alle": [r for r in records if r["category"] == "umgangssprachlich"],
        "Negativ alle": [r for r in records if not r["has_gold"]],
    }
    print(f"  {'Variante':<42} {'soll+rf':>7} {'umg.':>7} {'Safety':>7} {'Negativ leer':>13} {'Chunks':>7}")
    for name, fn, ms, win, k in GATES:
        sel = {id(r): simulate_gate(r["candidates"], fn, ms, win, k) for r in records}
        def hit(rows):
            return sum(1 for r in rows if any(c["is_gold"] for c in sel[id(r)]))
        soll = [r for r in records if r["category"] in ("soll", "rueckfrage")]
        umg = groups["umgangsspr. alle"]
        saf = [r for r in records if r["safety"]]
        neg = groups["Negativ alle"]
        empty = sum(1 for r in neg if not sel[id(r)])
        chunks = statistics.mean(len(sel[id(r)]) for r in records)
        print(
            f"  {name:<42} {hit(soll):>3}/{len(soll):<3} {hit(umg):>3}/{len(umg):<3} "
            f"{hit(saf):>3}/{len(saf):<3} {empty:>6}/{len(neg):<6} {chunks:>7.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", nargs="?", default=str(DEFAULT_OUT))
    parser.add_argument("--testset", default=str(TESTSET))
    parser.add_argument("--qdrant", default=None, help="anderer Qdrant-Ordner, z. B. data/qdrant_gross")
    args = parser.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    questions = yaml.safe_load(Path(args.testset).read_text(encoding="utf-8"))["questions"]
    config = load_config("config.yaml")
    if args.qdrant:
        config["rag"]["qdrant_path"] = args.qdrant
    rag = config["rag"]
    rr_cfg = rag["reranker"]
    min_score = rr_cfg["min_score"]
    window = rr_cfg.get("score_window", 1.0)
    final_k = rag["final_k"]

    retriever = KnowledgeRetriever(config)
    reranker = KnowledgeReranker(
        model_name=rr_cfg["model"], device=rr_cfg["device"]
    )

    points, _ = retriever._client.scroll(
        collection_name=rag["collection"],
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )
    meta_by_hash = {}
    for point in points:
        payload = point.payload or {}
        meta_by_hash[payload.get("content_hash")] = {
            "source": payload.get("source"),
            "headings": (payload.get("docling") or {}).get("headings"),
        }

    retriever.warmup()
    reranker.rerank("warmup", retriever.search("warmup"))

    records = []
    for item in questions:
        question = build_retrieval_query(
            [*item.get("history", []), {"role": "user", "content": item["q"]}]
        )
        gold = {tuple(g) for g in item.get("gold", [])}

        t0 = time.perf_counter()
        hits = retriever.search(question)
        t1 = time.perf_counter()
        reranker.rerank(question, hits)  # produktiver Aufruf, nur Zeitmessung
        t2 = time.perf_counter()

        raw = reranker._model.predict([(question, h.text) for h in hits])
        emb = reranker._model.predict([(question, h.embedding_text) for h in hits])

        candidates = []
        for hit, r, e in zip(hits, raw, emb):
            meta = meta_by_hash.get(hit.content_hash, {})
            candidates.append(
                {
                    "source": meta.get("source"),
                    "heading": key(meta)[1],
                    "raw": float(r),
                    "emb": float(e),
                    "is_gold": key(meta) in gold,
                }
            )

        ranked = sorted(
            candidates,
            key=lambda c: VARIANTS["avg(raw,emb) [aktuell]"](c["raw"], c["emb"]),
            reverse=True,
        )
        selected = []
        if ranked:
            top = VARIANTS["avg(raw,emb) [aktuell]"](ranked[0]["raw"], ranked[0]["emb"])
            if top >= min_score:
                floor = max(min_score, top - window)
                selected = [
                    c
                    for c in ranked
                    if VARIANTS["avg(raw,emb) [aktuell]"](c["raw"], c["emb"]) >= floor
                ][:final_k]

        records.append(
            {
                "question": question,
                "category": item["category"],
                "set": item.get("set", "e001"),
                "safety": bool(item.get("safety")),
                "has_gold": bool(gold),
                "retrieval_ms": round((t1 - t0) * 1000, 1),
                "rerank_ms": round((t2 - t1) * 1000, 1),
                "gold_in_candidates": any(c["is_gold"] for c in candidates),
                "selected_n": len(selected),
                "gold_in_selected": any(c["is_gold"] for c in selected),
                "candidates": candidates,
            }
        )

    positives = [r for r in records if r["has_gold"]]
    negatives = [r for r in records if not r["has_gold"]]

    print(
        f"Testset: {len(records)} Fragen ({len(positives)} mit Gold, "
        f"{len(negatives)} Negativ) | Gate: min_score={min_score}, "
        f"window={window}, final_k={final_k}, "
        f"candidate_k={rag['candidate_k']}"
    )

    print("\n== Ranking (Rang des ersten Gold-Chunks unter den Kandidaten) ==")
    variant_summary = {}
    for name, fn in VARIANTS.items():
        ranks = []
        for r in positives:
            order = sorted(
                r["candidates"], key=lambda c: fn(c["raw"], c["emb"]), reverse=True
            )
            pos = next((i + 1 for i, c in enumerate(order) if c["is_gold"]), None)
            ranks.append(pos)
        top1 = sum(1 for p in ranks if p == 1)
        top3 = sum(1 for p in ranks if p is not None and p <= 3)
        variant_summary[name] = {"top1": top1, "top3": top3, "ranks": ranks}
        print(f"  {name:<24} Rang1 {top1}/{len(positives)}  Top3 {top3}/{len(positives)}")

    print("\n== Produktiv-Gate: was sieht das LLM wirklich? ==")
    cats = ["soll", "umgangssprachlich", "rueckfrage", "nicht_im_wissen", "kein_wissen"]
    for cat in cats:
        rows = [r for r in records if r["category"] == cat]
        if not rows:
            continue
        if rows[0]["has_gold"]:
            ok = sum(1 for r in rows if r["gold_in_selected"])
            empty = sum(1 for r in rows if r["selected_n"] == 0)
            print(
                f"  {cat:<18} Gold im Kontext {ok}/{len(rows)}  "
                f"(davon Kontext leer: {empty})"
            )
        else:
            clean = sum(1 for r in rows if r["selected_n"] == 0)
            print(f"  {cat:<18} korrekt leer {clean}/{len(rows)}")

    print(
        "  Recall@candidate_k (Gold überhaupt unter den Kandidaten): "
        f"{sum(1 for r in positives if r['gold_in_candidates'])}/{len(positives)}"
    )

    safety = [r for r in records if r["safety"]]
    print("\n== Safety-Fragen ==")
    for r in safety:
        print(
            f"  Gold im Kontext: {'ja ' if r['gold_in_selected'] else 'NEIN'} | "
            f"selected={r['selected_n']} | {r['question']}"
        )

    print_gate_comparison(records, min_score, window, final_k)

    ret = [r["retrieval_ms"] for r in records]
    rer = [r["rerank_ms"] for r in records]
    print("\n== Latenz (ms, CPU, nach Warmup) ==")
    print(
        f"  Retrieval  Median {statistics.median(ret):.0f}  Max {max(ret):.0f}\n"
        f"  Rerank     Median {statistics.median(rer):.0f}  Max {max(rer):.0f}"
    )

    out_path.write_text(
        json.dumps(
            {
                "date": date.today().isoformat(),
                "config": {
                    "embedding_model": rag["embedding_model"],
                    "sparse_model": rag["sparse_model"],
                    "reranker": rr_cfg,
                    "candidate_k": rag["candidate_k"],
                    "final_k": final_k,
                },
                "chunk_count": len(points),
                "variants": variant_summary,
                "records": records,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\nErgebnis gespeichert: {out_path}")
    retriever.close()


if __name__ == "__main__":
    main()

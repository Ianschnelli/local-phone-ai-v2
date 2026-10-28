"""Vergleich von Embedding-Modellen für die Suche (E-015), nur Messung.

Liest die Abschnitte (embedding_text) aus einem bestehenden Qdrant-Index,
bettet sie mit jedem Modell im Speicher ein und misst, ob der Soll-Abschnitt
unter den Top-k landet: dense allein und hybrid mit BM25 (RRF wie in der App).
Ändert keinen Index.

Aufruf:
    .venv\\Scripts\\python.exe eval\\embedding_compare.py
        [--testset eval/rag_testset_gross.yaml] [--qdrant data/qdrant_gross]
        [--models minilm,e5-large,...] [--field embedding_text]
"""
import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("FASTEMBED_CACHE_PATH", str(Path("models/fastembed").resolve()))

import numpy as np
import yaml
from loguru import logger

logger.remove()
logger.add(sys.stderr, level="WARNING")

from qdrant_client import QdrantClient

from local_phone_ai_v2.knowledge.query_builder import build_retrieval_query

# Name -> (Bibliothek, Modell, Präfix Anfrage, Präfix Abschnitt)
MODELS = {
    "minilm": ("fastembed", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "", ""),
    "mpnet": ("fastembed", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "", ""),
    "jina-de": ("fastembed", "jinaai/jina-embeddings-v2-base-de", "", ""),
    "e5-large": ("fastembed", "intfloat/multilingual-e5-large", "query: ", "passage: "),
    "jina-v3": ("fastembed", "jinaai/jina-embeddings-v3", "", ""),
    "bge-m3": ("st", "BAAI/bge-m3", "", ""),
    # dieselben großen Modelle über sentence-transformers (GPU, falls vorhanden)
    "e5-large-st": ("st", "intfloat/multilingual-e5-large", "query: ", "passage: "),
    "mpnet-st": ("st", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "", ""),
}
KS = (1, 3, 5, 8, 16)


def chunk_key(payload: dict) -> tuple[str, str]:
    headings = (payload.get("docling") or {}).get("headings") or [""]
    return payload.get("source", ""), headings[-1]


def make_encoder(name: str):
    lib, model_name, q_prefix, p_prefix = MODELS[name]
    if lib == "fastembed":
        from fastembed import TextEmbedding

        model = TextEmbedding(model_name)

        def encode(texts, is_query):
            prefix = q_prefix if is_query else p_prefix
            return np.array(list(model.embed([prefix + t for t in texts])))
    else:
        from sentence_transformers import SentenceTransformer

        from local_phone_ai_v2.config.devices import resolve_torch_device

        model = SentenceTransformer(model_name, device=resolve_torch_device("auto"))

        def encode(texts, is_query):
            prefix = q_prefix if is_query else p_prefix
            return model.encode([prefix + t for t in texts], normalize_embeddings=True)
    return encode


def normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def bm25_scores(queries: list[str], docs: list[str]) -> np.ndarray:
    from fastembed import SparseTextEmbedding

    model = SparseTextEmbedding("Qdrant/bm25")
    doc_vecs = list(model.embed(docs))
    query_vecs = list(model.query_embed(queries))
    scores = np.zeros((len(queries), len(docs)))
    for qi, qv in enumerate(query_vecs):
        q = dict(zip(qv.indices.tolist(), qv.values.tolist()))
        for di, dv in enumerate(doc_vecs):
            scores[qi, di] = sum(q.get(i, 0.0) * v for i, v in zip(dv.indices.tolist(), dv.values.tolist()))
    return scores


def rrf(score_lists: list[np.ndarray], k: int = 60, depth: int = 16) -> np.ndarray:
    """Reciprocal Rank Fusion über die jeweils besten `depth` Treffer (wie Qdrant-Prefetch)."""
    fused = np.zeros_like(score_lists[0])
    for scores in score_lists:
        order = np.argsort(-scores, axis=1)[:, :depth]
        for qi in range(scores.shape[0]):
            for rank, di in enumerate(order[qi]):
                fused[qi, di] += 1.0 / (k + rank + 1)
    return fused


def recall_table(scores: np.ndarray, gold_sets: list[set[int]], idx: list[int]) -> dict[int, int]:
    order = np.argsort(-scores, axis=1)
    return {k: sum(1 for i in idx if gold_sets[i] & set(order[i, :k].tolist())) for k in KS}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", default="eval/rag_testset_gross.yaml")
    parser.add_argument("--qdrant", default="data/qdrant_gross")
    parser.add_argument("--models", default=",".join(MODELS))
    parser.add_argument("--field", default="embedding_text")
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    client = QdrantClient(path=args.qdrant)
    points, _ = client.scroll(collection_name="knowledge", limit=5000, with_payload=True, with_vectors=False)
    client.close()
    docs = [p.payload.get(args.field) or p.payload["text"] for p in points]
    keys = [chunk_key(p.payload) for p in points]

    items = yaml.safe_load(Path(args.testset).read_text(encoding="utf-8"))["questions"]
    queries = [build_retrieval_query([*i.get("history", []), {"role": "user", "content": i["q"]}]) for i in items]
    gold_sets = [{d for d, key in enumerate(keys) if key in {tuple(g) for g in i.get("gold", [])}} for i in items]
    groups = {
        "soll+rf": [n for n, i in enumerate(items) if i.get("gold") and i["category"] in ("soll", "rueckfrage")],
        "umg.": [n for n, i in enumerate(items) if i.get("gold") and i["category"] == "umgangssprachlich"],
        "safety": [n for n, i in enumerate(items) if i.get("gold") and i.get("safety")],
    }

    print(f"{len(docs)} Abschnitte ({args.field}) | {len(items)} Fragen | "
          + " ".join(f"{g}={len(v)}" for g, v in groups.items()))
    sparse = bm25_scores(queries, docs)

    summary = {}
    header = "  {:<10} {:<7} {:<7}".format("Modell", "Modus", "Gruppe") + "".join(f"{'R@' + str(k):>8}" for k in KS)
    for name in args.models.split(","):
        t0 = time.perf_counter()
        encode = make_encoder(name)
        doc_emb = normalize(np.asarray(encode(docs, False), dtype=np.float32))
        t1 = time.perf_counter()
        q_times = []
        q_emb = []
        for q in queries:
            s = time.perf_counter()
            q_emb.append(np.asarray(encode([q], True), dtype=np.float32)[0])
            q_times.append((time.perf_counter() - s) * 1000)
        dense = normalize(np.array(q_emb)) @ doc_emb.T
        hybrid = rrf([dense, sparse])
        print(f"\n== {name}: Index {t1 - t0:.0f} s, Anfrage-Embedding Median {statistics.median(q_times):.0f} ms")
        print(header)
        summary[name] = {"query_ms": statistics.median(q_times)}
        for mode, scores in (("dense", dense), ("hybrid", hybrid)):
            for group, idx in groups.items():
                table = recall_table(scores, gold_sets, idx)
                summary[name][f"{mode}/{group}"] = table
                print(f"  {name:<10} {mode:<7} {group:<7}"
                      + "".join(f"{table[k]:>5}/{len(idx):<2}" for k in KS))

    today = datetime.now().astimezone().date().isoformat()
    out = Path(f"docs/rag/daten/{today}_embedding_compare{('_' + args.label) if args.label else ''}.json")
    out.write_text(json.dumps({"field": args.field, "qdrant": args.qdrant, "summary": summary},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nRohdaten: {out}")


if __name__ == "__main__":
    main()

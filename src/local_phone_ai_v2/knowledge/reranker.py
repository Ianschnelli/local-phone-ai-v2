from loguru import logger
from sentence_transformers import CrossEncoder

from local_phone_ai_v2.config.devices import resolve_torch_device

from .retriever import KnowledgeHit


class KnowledgeReranker:
    """Bewertet die Kandidaten der Suche gegen die Anfrage (bge-reranker-v2-m3).

    Ein Durchlauf über den Suchtext (Überschriften + Abschnitt + Anrufer-
    formulierungen der Anreicherung), E-015. fp16 nur auf der GPU: gleiche
    Reihenfolge bei ca. dreifachem Tempo (E-018).
    """

    def __init__(
        self,
        model_name: str,
        device: str,
        precision: str = "fp32",
    ) -> None:
        resolved_device = resolve_torch_device(device)
        model_kwargs = {}

        if precision == "fp16" and resolved_device == "cuda":
            import torch

            model_kwargs["torch_dtype"] = torch.float16

        self._model = CrossEncoder(
            model_name,
            device=resolved_device,
            local_files_only=True,
            model_kwargs=model_kwargs,
        )

    def rerank(
        self,
        query: str,
        hits: list[KnowledgeHit],
    ) -> list[KnowledgeHit]:
        if not hits:
            return []

        scores = self._model.predict(
            [(query, hit.embedding_text) for hit in hits]
        )
        ranked = sorted(
            (
                KnowledgeHit(
                    text=hit.text,
                    embedding_text=hit.embedding_text,
                    score=float(score),
                    content_hash=hit.content_hash,
                )
                for hit, score in zip(hits, scores)
            ),
            key=lambda hit: hit.score,
            reverse=True,
        )

        logger.debug(
            "RAG reranking: {}",
            [
                {"rank": index, "score": round(hit.score, 4), "preview": hit.text[:120]}
                for index, hit in enumerate(ranked, start=1)
            ],
        )

        return ranked

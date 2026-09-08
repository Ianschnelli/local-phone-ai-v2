from sentence_transformers import CrossEncoder

from .retriever import KnowledgeHit
from loguru import logger

class KnowledgeReranker:
    def __init__(
        self,
        model_name: str,
        device: str,
    ) -> None:
        self._model = CrossEncoder(
            model_name,
            device=device,
            local_files_only=True,
        )

    def rerank(
        self,
        query: str,
        hits: list[KnowledgeHit],
    ) -> list[KnowledgeHit]:
        if not hits:
            return []

        pairs = [
            (query, hit.text)
            for hit in hits
        ]

        scores = self._model.predict(
            pairs
        )

        ranked = sorted(
            zip(hits, scores),
            key=lambda item: float(item[1]),
            reverse=True,
        )

        logger.debug(
            "RAG reranking: {}",
            [
                {
                    "rank": index,
                    "score": round(float(score), 4),
                    "preview": hit.text[:120],
                }
                for index, (hit, score) in enumerate(
                    ranked,
                    start=1,
                )
            ],
        )

        return [
            KnowledgeHit(
                text=hit.text,
                score=float(score),
                content_hash=hit.content_hash,
            )
            for hit, score in ranked
        ]
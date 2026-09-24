from loguru import logger
from sentence_transformers import CrossEncoder

from .retriever import KnowledgeHit


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

        raw_pairs = [
            (
                query,
                hit.text,
            )
            for hit in hits
        ]

        embedding_pairs = [
            (
                query,
                hit.embedding_text,
            )
            for hit in hits
        ]

        raw_scores = self._model.predict(
            raw_pairs
        )

        embedding_scores = self._model.predict(
            embedding_pairs
        )

        scored = [
            (
                hit,
                float(raw_score),
                float(embedding_score),
                (
                    float(raw_score)
                    + float(embedding_score)
                ) / 2,
            )
            for (
                hit,
                raw_score,
                embedding_score,
            ) in zip(
                hits,
                raw_scores,
                embedding_scores,
            )
        ]

        ranked = sorted(
            scored,
            key=lambda item: item[3],
            reverse=True,
        )

        logger.debug(
            "RAG reranking: {}",
            [
                {
                    "rank": index,
                    "score": round(
                        combined_score,
                        4,
                    ),
                    "raw_score": round(
                        raw_score,
                        4,
                    ),
                    "embedding_score": round(
                        embedding_score,
                        4,
                    ),
                    "preview": hit.text[:120],
                }
                for index, (
                    hit,
                    raw_score,
                    embedding_score,
                    combined_score,
                ) in enumerate(
                    ranked,
                    start=1,
                )
            ],
        )

        return [
            KnowledgeHit(
                text=hit.text,
                embedding_text=hit.embedding_text,
                score=combined_score,
                content_hash=hit.content_hash,
            )
            for (
                hit,
                _raw_score,
                _embedding_score,
                combined_score,
            ) in ranked
        ]
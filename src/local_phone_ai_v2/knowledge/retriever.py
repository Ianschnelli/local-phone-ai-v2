from dataclasses import dataclass

from loguru import logger
from qdrant_client import QdrantClient, models


@dataclass(frozen=True)
class KnowledgeHit:
    text: str
    score: float
    content_hash: str


class KnowledgeRetriever:
    def __init__(self, config: dict) -> None:
        rag_config = config["rag"]

        self._client = QdrantClient(
            path=rag_config["qdrant_path"],
        )

        self._collection = rag_config["collection"]
        self._dense_model = rag_config["embedding_model"]
        self._sparse_model = rag_config["sparse_model"]

        self._candidate_k = rag_config[
            "candidate_k"
        ]

    def search(
        self,
        query: str,
    ) -> list[KnowledgeHit]:
        if not query.strip():
            return []

        try:
            result = self._client.query_points(
                collection_name=self._collection,
                prefetch=[
                    models.Prefetch(
                        query=models.Document(
                            text=query,
                            model=self._dense_model,
                        ),
                        using="dense",
                        limit=self._candidate_k,
                    ),
                    models.Prefetch(
                        query=models.Document(
                            text=query,
                            model=self._sparse_model,
                        ),
                        using="sparse",
                        limit=self._candidate_k,
                    ),
                ],
                query=models.FusionQuery(
                    fusion=models.Fusion.RRF,
                ),
                limit=self._candidate_k,
                with_payload=True,
            )

        except Exception:
            logger.exception(
                "RAG retrieval failed."
            )
            return []

        hits: list[KnowledgeHit] = []
        seen: set[str] = set()

        for point in result.points:
            if not point.payload:
                continue

            text = point.payload.get("text")

            if not isinstance(text, str):
                continue

            text = text.strip()

            if not text:
                continue

            content_hash = point.payload.get(
                "content_hash"
            )

            if not isinstance(
                content_hash,
                str,
            ):
                content_hash = text

            if content_hash in seen:
                continue

            seen.add(content_hash)

            hits.append(
                KnowledgeHit(
                    text=text,
                    score=float(point.score),
                    content_hash=content_hash,
                )
            )

        logger.debug(
            "RAG retrieval: {} unique candidate(s): {}",
            len(hits),
            [
                {
                    "rank": index + 1,
                    "score": round(hit.score, 4),
                    "preview": (
                        hit.text
                        .replace("\n", " ")
                        [:120]
                    ),
                }
                for index, hit in enumerate(hits)
            ],
        )

        return hits

    def warmup(self) -> None:
        if not self._client.collection_exists(
            self._collection
        ):
            raise RuntimeError(
                "RAG collection not found: "
                f"{self._collection}"
            )

        self._client.query_points(
            collection_name=self._collection,
            prefetch=[
                models.Prefetch(
                    query=models.Document(
                        text="warmup",
                        model=self._dense_model,
                    ),
                    using="dense",
                    limit=1,
                ),
                models.Prefetch(
                    query=models.Document(
                        text="warmup",
                        model=self._sparse_model,
                    ),
                    using="sparse",
                    limit=1,
                ),
            ],
            query=models.FusionQuery(
                fusion=models.Fusion.RRF,
            ),
            limit=1,
            with_payload=False,
        )

    def close(self) -> None:
        self._client.close()
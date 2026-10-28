"""Dense-Embeddings für Einlesen und Suche (E-015).

Zwei Wege, per ``rag.embedding_backend`` gewählt:
- ``fastembed`` (bisher): Qdrant rechnet die Vektoren selbst aus dem
  Modellnamen (``models.Document``). Leicht, läuft gut auf der CPU.
- ``sentence_transformers``: wir rechnen die Vektoren selbst, z. B. für
  ``BAAI/bge-m3``, das fastembed nicht kennt. Gerät über ``device: auto``.
"""
from __future__ import annotations

from qdrant_client import QdrantClient, models

from local_phone_ai_v2.config.devices import resolve_torch_device


class DenseEmbedder:
    def __init__(
        self,
        rag_config: dict,
        local_files_only: bool = True,
    ) -> None:
        # On-Prem: zur Laufzeit nur von der Festplatte laden; herunterladen
        # darf nur setup (local_files_only=False).
        self._backend = rag_config.get("embedding_backend", "fastembed")
        self._model_name = rag_config["embedding_model"]
        self._query_prefix = rag_config.get("embedding_query_prefix", "")
        self._passage_prefix = rag_config.get("embedding_passage_prefix", "")
        self._model = None

        if self._backend == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            device = resolve_torch_device(
                rag_config.get("embedding_device", "auto")
            )
            model_kwargs = {}

            # fp16 nur auf der GPU: halber Grafikspeicher (E-019).
            if rag_config.get("embedding_precision", "fp32") == "fp16" and device == "cuda":
                import torch

                model_kwargs["torch_dtype"] = torch.float16

            self._model = SentenceTransformer(
                self._model_name,
                device=device,
                model_kwargs=model_kwargs,
                local_files_only=local_files_only,
            )
        elif self._backend != "fastembed":
            raise ValueError(
                f"Unsupported embedding_backend: {self._backend}. "
                "Use 'fastembed' or 'sentence_transformers'."
            )

    def size(
        self,
        client: QdrantClient,
    ) -> int:
        if self._model is None:
            return client.get_embedding_size(self._model_name)

        return self._model.get_embedding_dimension()

    def query_vector(
        self,
        text: str,
    ) -> models.Document | list[float]:
        if self._model is None:
            return models.Document(text=text, model=self._model_name)

        return self._model.encode(
            self._query_prefix + text,
            normalize_embeddings=True,
        ).tolist()

    def document_vectors(
        self,
        texts: list[str],
    ) -> list[models.Document | list[float]]:
        if self._model is None:
            return [
                models.Document(text=text, model=self._model_name)
                for text in texts
            ]

        return self._model.encode(
            [self._passage_prefix + text for text in texts],
            normalize_embeddings=True,
            batch_size=16,
        ).tolist()

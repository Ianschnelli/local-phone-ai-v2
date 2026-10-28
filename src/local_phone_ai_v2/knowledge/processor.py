"""Wissen pro Gesprächsschritt einspeisen, schlank nach Best Practice (E-023).

Ein Gesprächsschritt:
1. Notfall-Warnzeichen → feste Ansage, ohne LLM (safety.py).
2. Suche mit der Anruferäußerung (bei kurzen Folgefragen plus der vorigen).
3. Eine Relevanzschwelle, höchstens final_k Abschnitte.
4. Ein Wissensblock als eigene Nachricht nach der Anruferfrage, mit einem
   Satz Regel. Ohne passenden Treffer kommt kein Block.
"""
import asyncio
import threading

from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMContextFrame,
    TTSSpeakFrame,
)
from pipecat.processors.aggregators.llm_context import (
    LLMContext,
)
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
)

from local_phone_ai_v2.knowledge.query_builder import (
    build_retrieval_query,
)
from local_phone_ai_v2.knowledge.retriever import (
    KnowledgeHit,
    KnowledgeRetriever,
)

from .enrichment import ENRICHMENT_PREFIXES
from .emergency_gate import EmergencyInputGate
from .reranker import KnowledgeReranker
from .safety import emergency_message

KNOWLEDGE_RULE = (
    "Beantworte die Frage mit diesem Wissen, auch wenn der Anrufer andere "
    "Wörter benutzt; Öffnungszeiten nennst du aus den festen Daten. Nur wenn es die "
    "Frage nicht beantwortet, sage: \"Dazu liegt mir leider keine "
    "Information vor.\""
)

# Gemeinsame Modelle (Embedding, Reranker, Qdrant lokal) nie gleichzeitig
# nutzen: Ein abgelaufener Timeout lässt den Thread weiterlaufen (Review E-022).
_MODEL_LOCK = threading.Lock()


class KnowledgeContextProcessor(FrameProcessor):
    def __init__(
        self,
        retriever: KnowledgeRetriever,
        reranker: KnowledgeReranker,
        timeout_secs: float,
        final_k: int,
        min_score: float,
        emergency_gate: EmergencyInputGate | None = None,
    ) -> None:
        super().__init__()

        self._retriever = retriever
        self._reranker = reranker
        self._timeout_secs = timeout_secs
        self._final_k = final_k
        self._min_score = min_score
        self._emergency_gate = emergency_gate

    async def process_frame(
        self,
        frame: Frame,
        direction: FrameDirection,
    ) -> None:
        await super().process_frame(frame, direction)

        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        messages = frame.context.get_messages()
        last = messages[-1] if messages else None

        if (
            not isinstance(last, dict)
            or last.get("role") != "user"
            or not isinstance(last.get("content"), str)
            or not last["content"].strip()
        ):
            await self.push_frame(frame, direction)
            return

        user_text = last["content"].strip()

        # 1. Notfall: feste Ansage statt LLM (Vapi: "server-side mechanisms").
        notice = emergency_message(user_text)

        if notice:
            logger.warning("Emergency warning sign detected, fixed notice spoken.")

            if self._emergency_gate is not None:
                self._emergency_gate.activate()
            # Der Assistant-Aggregator schreibt die Ansage selbst in den
            # Kontext (append_to_context, wie bei der Begrüßung).
            await self.push_frame(
                TTSSpeakFrame(
                    text=notice,
                    append_to_context=True,
                ),
                direction,
            )
            return

        # 2./3. Suche, eine Schwelle, höchstens final_k Abschnitte.
        hits = await self._find(build_retrieval_query(messages))

        if not hits:
            await self.push_frame(frame, direction)
            return

        # 4. Ein Wissensblock nach der Anruferfrage; das Gespräch bleibt
        # unverändert, damit Rückfragen an der vorigen Antwort hängen (E-018).
        block = (
            "WISSEN ZUR FRAGE:\n"
            + "\n\n".join(self._display_text(hit) for hit in hits)
            + f"\n\n{KNOWLEDGE_RULE}"
        )
        augmented = [dict(message) for message in messages]
        augmented.append({"role": "system", "content": block})

        await self.push_frame(
            LLMContextFrame(context=LLMContext(messages=augmented)),
            direction,
        )

    async def _find(
        self,
        query: str,
    ) -> list[KnowledgeHit]:
        if not query:
            return []

        try:
            hits = await asyncio.wait_for(
                asyncio.to_thread(self._search_and_rerank, query),
                timeout=self._timeout_secs,
            )
        except Exception:  # noqa: BLE001
            # Timeout oder Modellfehler (z. B. Grafikspeicher): ohne Wissen
            # weiter, statt den Anrufer stumm zu lassen.
            logger.exception("RAG search failed, answering without knowledge.")
            return []

        return [hit for hit in hits if hit.score >= self._min_score][:self._final_k]

    def _search_and_rerank(
        self,
        query: str,
    ) -> list[KnowledgeHit]:
        with _MODEL_LOCK:
            return self._reranker.rerank(query, self._retriever.search(query))

    @staticmethod
    def _display_text(
        hit: KnowledgeHit,
    ) -> str:
        """Überschriften-Pfad plus Abschnitt, ohne die Anruferformulierungen
        der Anreicherung (E-018: Überschriften helfen Qwen beim Zuordnen)."""
        return "\n".join(
            line
            for line in hit.embedding_text.splitlines()
            if not line.startswith(ENRICHMENT_PREFIXES)
        )

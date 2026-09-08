from dataclasses import dataclass

from transformers import AutoTokenizer

from local_phone_ai_v2.conversation.summary import (
    CallSummarizer,
    CallSummary,
)

from loguru import logger


@dataclass
class ContextState:
    summary: dict | None = None
    compaction_count: int = 0
    escalation_required: bool = False


class ConversationContextManager:
    def __init__(
        self,
        tokenizer_model: str,
        context_window: int,
        response_reserve_tokens: int,
        compact_at_ratio: float,
        recent_turn_pairs: int,
        max_compactions: int,
        summarizer: CallSummarizer,
        system_instruction: str,
        rag_reserve_tokens: int,
    ) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model
        )

        self._context_window = context_window
        self._response_reserve_tokens = (
            response_reserve_tokens
        )
        self._compact_at_ratio = compact_at_ratio
        self._recent_turn_pairs = recent_turn_pairs
        self._max_compactions = max_compactions
        self._summarizer = summarizer
        self._system_instruction = system_instruction
        self._rag_reserve_tokens = rag_reserve_tokens
        self.state = ContextState()

    def count_tokens(
        self,
        messages: list[dict],
    ) -> int:
        encoded = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )

        return len(encoded["input_ids"])


    def should_compact(
        self,
        messages: list[dict],
    ) -> bool:
        keep_count = self._recent_turn_pairs * 2

        if len(messages) <= keep_count:
            return False

        old_messages = messages[:-keep_count]

        has_user_content = any(
            message.get("role") == "user"
            and isinstance(message.get("content"), str)
            and message["content"].strip()
            for message in old_messages
        )

        if not has_user_content:
            return False

        conversation_tokens = self.count_tokens(
            messages
        )

        system_tokens = self.count_tokens(
            [
                {
                    "role": "system",
                    "content": self._system_instruction,
                }
            ]
        )

        fixed_tokens = (
            system_tokens
            + self._rag_reserve_tokens
            + self._response_reserve_tokens
        )

        available_conversation_tokens = (
            self._context_window
            - fixed_tokens
        )

        threshold = int(
            available_conversation_tokens
            * self._compact_at_ratio
        )

        logger.debug(
            "Conversation context budget: "
            "conversation_tokens={}, "
            "system_tokens={}, "
            "rag_reserve_tokens={}, "
            "response_reserve_tokens={}, "
            "threshold={}, "
            "context_window={}, "
            "compaction_count={}",
            conversation_tokens,
            system_tokens,
            self._rag_reserve_tokens,
            self._response_reserve_tokens,
            threshold,
            self._context_window,
            self.state.compaction_count,
        )

        return conversation_tokens >= threshold

    def can_compact(self) -> bool:
        return (
            self.state.compaction_count
            < self._max_compactions
        )

    def prepare_compaction(
        self,
        messages: list[dict],
    ) -> tuple[CallSummary, list[dict]]:
        if not self.can_compact():
            raise RuntimeError(
                "Maximum conversation compactions reached."
            )

        keep_count = self._recent_turn_pairs * 2

        if len(messages) <= keep_count:
            raise RuntimeError(
                "Not enough messages to compact."
            )

        old_messages = messages[:-keep_count]
        recent_messages = messages[-keep_count:]

        summary = self._summarizer.summarize(
            old_messages
        )

        return summary, recent_messages


    def commit_compaction(
        self,
        summary: CallSummary,
    ) -> None:
        self.state.summary = summary.model_dump()
        self.state.compaction_count += 1

    def build_summary_message(
        self,
    ) -> dict | None:
        if self.state.summary is None:
            return None

        summary = CallSummary.model_validate(
            self.state.summary
        )

        return {
            "role": "developer",
            "content": (
                "STRUKTURIERTE ZUSAMMENFASSUNG "
                "DES BISHERIGEN GESPRÄCHS:\n"
                f"{summary.model_dump_json()}"
            ),
        }
    
    def requires_escalation(
        self,
        messages: list[dict],
    ) -> bool:
        if not self.should_compact(
            messages
        ):
            return False

        return not self.can_compact()

    def mark_escalation_required(
        self,
    ) -> None:
        self.state.escalation_required = True
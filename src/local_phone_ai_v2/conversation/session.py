import asyncio
from uuid import uuid4

from loguru import logger

from local_phone_ai_v2.conversation.context_manager import (
    ConversationContextManager,
)
from local_phone_ai_v2.pipeline.build import build_pipeline
from local_phone_ai_v2.pipeline.runner import run_pipeline
from local_phone_ai_v2.runtime.resources import RuntimeResources


class CallSession:
    def __init__(
        self,
        config: dict,
        transport,
        customer_instructions: str,
        greeting: str,
        resources: RuntimeResources,
    ) -> None:
        self.id = str(uuid4())
        self._greeting = greeting

        built = build_pipeline(
            config,
            transport,
            customer_instructions,
            resources,
        )

        self._pipeline = built.pipeline
        self._context = built.context
        self._user_aggregator = built.user_aggregator
        self._assistant_aggregator = built.assistant_aggregator

        conversation_config = config["conversation"]
        llm_config = config["llm"]

        self.context_manager = ConversationContextManager(
            tokenizer_model=llm_config["tokenizer_model"],
            context_window=llm_config["context_window"],
            response_reserve_tokens=llm_config[
                "response_reserve_tokens"
            ],
            compact_at_ratio=conversation_config[
                "compact_at_ratio"
            ],
            recent_turn_pairs=conversation_config[
                "recent_turn_pairs"
            ],
            max_compactions=conversation_config[
                "max_compactions"
            ],
            summarizer=resources.summarizer,
            system_instruction=built.system_instruction,
            rag_reserve_tokens=conversation_config[
                "rag_reserve_tokens"
            ],
        )

        self._compaction_task: asyncio.Task | None = None

        self._register_context_handlers()

    def _register_context_handlers(
        self,
    ) -> None:
        @self._assistant_aggregator.event_handler(
            "on_assistant_turn_stopped"
        )
        async def on_assistant_turn_stopped(
            aggregator,
            message,
        ) -> None:
            await self._handle_assistant_turn_stopped()

    async def _handle_assistant_turn_stopped(
        self,
    ) -> None:
        if (
            self._compaction_task is not None
            and not self._compaction_task.done()
        ):
            return

        messages = self._context.get_messages()

        logger.debug(
            "Context debug: object_id={}, messages={}",
            id(self._context),
            messages,
        )

        if not messages:
            return

        if not self.context_manager.should_compact(
            messages
        ):
            return

        if not self.context_manager.can_compact():
            self.context_manager.mark_escalation_required()

            logger.warning(
                "Conversation context escalation required: "
                "maximum compactions reached."
            )
            return

        snapshot = [
            message.copy()
            for message in messages
        ]

        self._compaction_task = asyncio.create_task(
            self._compact_context(
                snapshot
            )
        )

    async def _compact_context(
        self,
        snapshot: list[dict],
    ) -> None:
        summary, recent_messages = await asyncio.to_thread(
            self.context_manager.prepare_compaction,
            snapshot,
        )

        current_messages = self._context.get_messages()

        if current_messages != snapshot:
            logger.debug(
                "Conversation context changed during "
                "compaction; discarding stale result."
            )
            return

        self.context_manager.commit_compaction(
            summary
        )

        summary_message = (
            self.context_manager.build_summary_message()
        )

        compacted_messages = []

        if summary_message is not None:
            compacted_messages.append(
                summary_message
            )

        compacted_messages.extend(
            recent_messages
        )

        self._context.set_messages(
            compacted_messages
        )

        logger.debug(
            "Conversation context compacted: "
            "summary_created={}, recent_messages={}, "
            "compaction_count={}",
            summary_message is not None,
            len(recent_messages),
            self.context_manager.state.compaction_count,
        )

    async def run(
        self,
    ) -> None:
        await run_pipeline(
            self._pipeline,
            self._greeting,
        )

from pipecat.frames.frames import TTSSpeakFrame
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.llm import OpenAILLMService


LLM_FALLBACK_TEXT = (
    "Entschuldigung, aktuell gibt es ein technisches Problem. "
    "Bitte versuchen Sie es später erneut."
)


class FallbackOpenAILLMService(OpenAILLMService):
    async def push_error(
        self,
        error_msg: str,
        exception: Exception | None = None,
        fatal: bool = False,
    ):
        # Fixed speech output without calling the LLM again.
        await self.push_frame(
            TTSSpeakFrame(LLM_FALLBACK_TEXT),
            FrameDirection.DOWNSTREAM,
        )

        # Preserve Pipecat's normal error reporting.
        await super().push_error(
            error_msg=error_msg,
            exception=exception,
            fatal=fatal,
        )


def create_llm(
    config: dict,
    system_instruction: str,
) -> OpenAILLMService:
    return FallbackOpenAILLMService(
        api_key=config.get(
            "api_key",
            "lm-studio",
        ),
        base_url=config["base_url"],
        settings=OpenAILLMService.Settings(
            model=config["model"],
            system_instruction=system_instruction,
            temperature=config["temperature"],
            top_p=config["top_p"],
            max_tokens=config["max_tokens"],
        ),
    )

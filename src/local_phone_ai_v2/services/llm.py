from pipecat.services.ollama.llm import OLLamaLLMService


def create_llm(
    config: dict,
    system_instruction: str,
) -> OLLamaLLMService:
    extra = {}

    if config.get("thinking") is False:
        extra["reasoning_effort"] = "none"

    return OLLamaLLMService(
        settings=OLLamaLLMService.Settings(
            model=config["model"],
            system_instruction=system_instruction,
            temperature=config["temperature"],
            top_p=config["top_p"],
            max_tokens=config["max_tokens"],
            extra=extra,
        ),
    )
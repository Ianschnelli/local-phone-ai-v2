from pipecat.services.openai.llm import OpenAILLMService


def create_llm(
    config: dict,
    system_instruction: str,
) -> OpenAILLMService:
    return OpenAILLMService(
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
from pipecat.services.whisper.stt import WhisperSTTService
from local_phone_ai_v2.config.languages import get_language



def create_stt(config: dict) -> WhisperSTTService:
    device = config["device"]
    compute_type = "int8" if device == "cpu" else "float16"

    return WhisperSTTService(
        device=device,
        compute_type=compute_type,
        settings=WhisperSTTService.Settings(
            model=config["model"],
            language=get_language(config["language"]),
        ),
    )
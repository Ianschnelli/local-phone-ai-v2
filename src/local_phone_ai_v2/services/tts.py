from pathlib import Path

from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.transcriptions.language import Language

from local_phone_ai_v2.config.languages import get_language
from local_phone_ai_v2.services.tts_normalizer import normalize_tts_text

from pipecat.services.tts_service import (
    TTSService,
)

from local_phone_ai_v2.services.piper_tts import (
    create_piper_tts,
)


class GermanKokoroTTSService(KokoroTTSService):
    def language_to_service_language(self, language: Language) -> str:
        if language == Language.DE:
            return "de"

        return super().language_to_service_language(language)

    async def run_tts(self, text: str, context_id: str):
        normalized_text = normalize_tts_text(text)

        async for frame in super().run_tts(
            normalized_text,
            context_id,
        ):
            yield frame

def create_tts(
        config: dict,
    ) -> TTSService:
        provider = config["provider"]

        if provider == "kokoro":
            return create_kokoro_tts(
                config
            )

        if provider == "piper":
            return create_piper_tts(
                config
            )

        raise ValueError(
            f"Unsupported TTS provider: {provider}"
        )

def create_kokoro_tts(config: dict) -> GermanKokoroTTSService:
    model_dir = Path("models/kokoro_martin")

    return GermanKokoroTTSService(
        model_path=str(model_dir / "kokoro-martin.onnx"),
        voices_path=str(model_dir / "voices-martin.npz"),
        settings=GermanKokoroTTSService.Settings(
            voice=config["voice"],
            language=get_language(config["language"]),
        ),
    )
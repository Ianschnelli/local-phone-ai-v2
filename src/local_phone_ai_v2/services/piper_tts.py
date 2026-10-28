from pathlib import Path

from pipecat.services.piper.tts import (
    PiperTTSService,
)

from local_phone_ai_v2.services.tts_normalizer import normalize_tts_text


class GermanPiperTTSService(PiperTTSService):
    async def run_tts(self, text: str, context_id: str):
        normalized_text = normalize_tts_text(text)

        async for frame in super().run_tts(
            normalized_text,
            context_id,
        ):
            yield frame


def create_piper_tts(
    config: dict,
) -> GermanPiperTTSService:
    use_cuda = (
        config.get("device", "cpu").lower()
        == "cuda"
    )

    return GermanPiperTTSService(
        download_dir=Path("models/piper"),
        use_cuda=use_cuda,
        settings=GermanPiperTTSService.Settings(
            voice=config["model"],
        ),
    )

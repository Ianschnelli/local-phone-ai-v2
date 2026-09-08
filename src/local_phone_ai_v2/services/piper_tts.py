from pathlib import Path

from pipecat.services.piper.tts import (
    PiperTTSService,
)


def create_piper_tts(
    config: dict,
) -> PiperTTSService:
    use_cuda = (
        config.get("device", "cpu").lower()
        == "cuda"
    )

    return PiperTTSService(
        download_dir=Path("models/piper"),
        use_cuda=use_cuda,
        settings=PiperTTSService.Settings(
            voice=config["model"],
        ),
    )
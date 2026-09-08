from dataclasses import dataclass


@dataclass(frozen=True)
class AudioConfig:
    transport_sample_rate: int
    processing_sample_rate: int


def get_audio_config(config: dict) -> AudioConfig:
    audio = config["audio"]

    return AudioConfig(
        transport_sample_rate=audio["transport_sample_rate"],
        processing_sample_rate=audio["processing_sample_rate"],
    )
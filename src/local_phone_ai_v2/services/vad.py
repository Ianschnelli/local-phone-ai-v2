from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams


def create_vad(config: dict) -> SileroVADAnalyzer:
    return SileroVADAnalyzer(
        params=VADParams(
            confidence=config["confidence"],
            start_secs=config["start_secs"],
            stop_secs=config["stop_secs"],
            min_volume=config["min_volume"],
        )
    )
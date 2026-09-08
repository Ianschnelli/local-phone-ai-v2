from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import (
    LocalSmartTurnAnalyzerV3,
)


def create_turn_analyzer(sample_rate: int) -> LocalSmartTurnAnalyzerV3:
    return LocalSmartTurnAnalyzerV3(sample_rate=sample_rate)
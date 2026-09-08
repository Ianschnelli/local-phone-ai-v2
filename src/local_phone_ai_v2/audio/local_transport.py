from pipecat.transports.local.audio import (
    LocalAudioTransport,
    LocalAudioTransportParams,
)


def create_local_transport() -> LocalAudioTransport:
    return LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
        )
    )
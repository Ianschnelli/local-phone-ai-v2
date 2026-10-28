
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.processors.frame_processor import (
    FrameDirection,
    FrameProcessor,
)


class EmergencyInputGate(FrameProcessor):
    """Block user input while an emergency announcement is playing."""

    def __init__(self) -> None:
        super().__init__()
        self._active = False
        self._speech_started = False

    @property
    def active(self) -> bool:
        return self._active

    def activate(self) -> None:
        self._active = True
        self._speech_started = False

    async def process_frame(
        self,
        frame: Frame,
        direction: FrameDirection,
    ) -> None:
        await super().process_frame(frame, direction)

        if self._active and direction == FrameDirection.UPSTREAM:
            if isinstance(frame, BotStartedSpeakingFrame):
                self._speech_started = True

            elif isinstance(frame, BotStoppedSpeakingFrame):
                if self._speech_started:
                    self._active = False
                    self._speech_started = False

        if self._active and direction == FrameDirection.DOWNSTREAM:
            if isinstance(
                frame,
                (
                    InputAudioRawFrame,
                    InterimTranscriptionFrame,
                    TranscriptionFrame,
                    VADUserStartedSpeakingFrame,
                    VADUserStoppedSpeakingFrame,
                    UserStartedSpeakingFrame,
                    UserStoppedSpeakingFrame,
                ),
            ):
                return

        await self.push_frame(frame, direction)

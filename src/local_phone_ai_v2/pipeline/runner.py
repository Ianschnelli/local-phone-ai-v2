from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask


async def run_pipeline(
    pipeline,
    greeting: str,
) -> None:
    task = PipelineTask(pipeline)
    runner = PipelineRunner()

    await task.queue_frame(
        TTSSpeakFrame(
            text=greeting,
            append_to_context=True,
        )
    )

    await runner.run(task)
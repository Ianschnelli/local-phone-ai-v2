from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.turns.user_stop import (
    SpeechTimeoutUserTurnStopStrategy,
    TurnAnalyzerUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from local_phone_ai_v2.conversation.instructions import build_instructions
from local_phone_ai_v2.services.llm import create_llm
from local_phone_ai_v2.services.stt import create_stt
from local_phone_ai_v2.services.tts import create_tts
from local_phone_ai_v2.services.turn import create_turn_analyzer
from local_phone_ai_v2.services.vad import create_vad
from local_phone_ai_v2.runtime.resources import RuntimeResources
from local_phone_ai_v2.knowledge.processor import KnowledgeContextProcessor
from dataclasses import dataclass

from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.aggregators.llm_context import LLMContext



@dataclass
class BuiltPipeline:
    pipeline: Pipeline
    context: LLMContext
    user_aggregator: object
    assistant_aggregator: object
    system_instruction: str

def build_pipeline(
    config: dict,
    transport,
    customer_instructions: str,
    resources: RuntimeResources,
) -> BuiltPipeline:

    stt = create_stt(config["stt"])

    instructions = build_instructions(
        customer_instructions= customer_instructions,
    )

    llm = create_llm(
        config["llm"],
        instructions,
    )

    tts = create_tts(config["tts"])

    vad = create_vad(config["vad"])

    turn = create_turn_analyzer(
        config["audio"]["processing_sample_rate"]
    )

    context = LLMContext()

    knowledge = KnowledgeContextProcessor(
        resources.knowledge,
        resources.reranker,
        resources.facts,
        timeout_secs=config[
            "rag"
        ]["query_timeout_secs"],
        reranker_timeout_secs=config[
            "rag"
        ]["reranker_timeout_secs"],
        final_k=config["rag"]["final_k"],

        min_reranker_score=(
            config["rag"]["reranker"]["min_score"]
        ),
        reranker_score_window=(
            config["rag"]["reranker"]["score_window"]
        ),
    )

    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=vad,
            user_turn_strategies=UserTurnStrategies(
                start=[
                    VADUserTurnStartStrategy(
                        enable_interruptions=True,
                    )
                ],
                stop=[
                    TurnAnalyzerUserTurnStopStrategy(
                        turn_analyzer=turn,
                    ),
                    SpeechTimeoutUserTurnStopStrategy(
                        user_speech_timeout=config["turn"][
                            "user_speech_timeout"
                        ],
                    ),
                ],
            ),
        ),
    )

    return BuiltPipeline(
        pipeline=Pipeline(
            [
                transport.input(),
                stt,
                user_aggregator,
                knowledge,
                llm,
                tts,
                transport.output(),
                assistant_aggregator,
            ]
        ),
        context=context,
        user_aggregator=user_aggregator,
        assistant_aggregator=assistant_aggregator,
        system_instruction=instructions,
    )
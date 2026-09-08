from local_phone_ai_v2.knowledge.reranker import (
    KnowledgeReranker,
)
from local_phone_ai_v2.knowledge.retriever import (
    KnowledgeRetriever,
)
from local_phone_ai_v2.conversation.summary import (
    CallSummarizer,
)

from pathlib import Path

from local_phone_ai_v2.knowledge.facts import (
    StructuredFactProvider,
)

class RuntimeResources:
    def __init__(self, config: dict) -> None:
        self.knowledge = KnowledgeRetriever(
            config
        )

        reranker_config = config["rag"]["reranker"]

        self.reranker = KnowledgeReranker(
            model_name=reranker_config["model"],
            device=reranker_config["device"],
        )

        facts_path = Path(
                    config["customer"]["facts_path"]
                )

        self.facts = StructuredFactProvider(
                    facts_path=facts_path,
                )

        
        self.summarizer = CallSummarizer(
            model=config["llm"]["model"],
            max_tokens=config["conversation"]["summary_max_tokens"],
        )

    def warmup(self) -> None:
        self.knowledge.warmup()

    def close(self) -> None:
        self.knowledge.close()
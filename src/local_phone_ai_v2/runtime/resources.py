from pathlib import Path

from local_phone_ai_v2.conversation.summary import (
    CallSummarizer,
)
from local_phone_ai_v2.knowledge.facts import (
    StructuredFactProvider,
)
from local_phone_ai_v2.knowledge.reranker import (
    KnowledgeReranker,
)
from local_phone_ai_v2.knowledge.retriever import (
    KnowledgeRetriever,
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
            precision=reranker_config.get("precision", "fp32"),
        )

        facts_path = Path(
                    config["customer"]["facts_path"]
                )

        self.facts = StructuredFactProvider(
                    facts_path=facts_path,
                )

        
        self.summarizer = CallSummarizer(
            base_url=config["llm"]["base_url"],
            api_key=config["llm"]["api_key"],
            model=config["llm"]["model"],
            max_tokens=config["conversation"]["summary_max_tokens"],
        )

    def warmup(self) -> None:
        self.knowledge.warmup()

    def close(self) -> None:
        self.knowledge.close()
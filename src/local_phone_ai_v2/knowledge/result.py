from dataclasses import dataclass, field


@dataclass
class KnowledgeResult:
    exact_facts: list[str] = field(
        default_factory=list
    )
    retrieved_knowledge: list[str] = field(
        default_factory=list
    )

    @property
    def has_knowledge(self) -> bool:
        return bool(
            self.exact_facts
            or self.retrieved_knowledge
        )
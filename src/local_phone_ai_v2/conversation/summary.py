from openai import OpenAI
from pydantic import BaseModel, Field


class CallSummary(BaseModel):
    anliegen: list[str] = Field(default_factory=list)
    genannte_fakten: list[str] = Field(default_factory=list)
    genannte_entitaeten: list[str] = Field(default_factory=list)
    bereits_geklaert: list[str] = Field(default_factory=list)
    offene_punkte: list[str] = Field(default_factory=list)
    ausgefuehrte_aktionen: list[str] = Field(default_factory=list)


class CallSummarizer:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        max_tokens: int,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens

        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def summarize(
        self,
        messages: list[dict],
    ) -> CallSummary:
        conversation = "\n".join(
            f"user: {message['content']}"
            for message in messages
            if (
                message.get("role") == "user"
                and isinstance(
                    message.get("content"),
                    str,
                )
            )
        )

        prompt = (
            "Erstelle eine strukturierte Gedächtnis-Zusammenfassung "
            "des Anrufers.\n\n"
            "Verwende ausschließlich Informationen, die der Anrufer "
            "selbst ausdrücklich genannt hat.\n"
            "Keine Vermutungen. Keine Ergänzungen. Kein Firmenwissen. "
            "Keine Informationen aus früheren Antworten der Assistentin.\n"
            "Erfinde keine ausgeführten Aktionen.\n"
            "Wenn für ein Feld nichts sicher bekannt ist, "
            "verwende eine leere Liste.\n\n"
            f"AUSSAGEN DES ANRUFERS:\n{conversation}"
        )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0,
            max_tokens=self._max_tokens,
            stream=False,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "call_summary",
                    "strict": True,
                    "schema": CallSummary.model_json_schema(),
                },
            },
        )

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError(
                "LM Studio returned an empty call summary."
            )

        return CallSummary.model_validate_json(
            content
        )
import json
from urllib import request

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
        model: str,
        max_tokens: int,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens

    def summarize(
        self,
        messages: list[dict],
    ) -> CallSummary:
        conversation = "\n".join(
            f"user: {message['content']}"
            for message in messages
            if (
                message.get("role") == "user"
                and isinstance(message.get("content"), str)
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

        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "stream": False,
            "format": CallSummary.model_json_schema(),
            "options": {
                "temperature": 0,
                "num_predict": self._max_tokens,
            },
            "keep_alive": "30m",
        }

        data = json.dumps(
            payload
        ).encode("utf-8")

        req = request.Request(
            "http://localhost:11434/api/chat",
            data=data,
            headers={
                "Content-Type": "application/json"
            },
            method="POST",
        )

        with request.urlopen(
            req,
            timeout=120,
        ) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

        content = result["message"]["content"]

        return CallSummary.model_validate_json(
            content
        )
"""Anreicherung von Wissensabschnitten beim Einlesen (E-016).

Qwen schreibt einmalig pro Abschnitt, worum es geht und wie Anrufer am
Telefon danach fragen würden (umgangssprachlich, mit Alltagswörtern und
Symptomen statt Fachbegriffen). Das landet nur im Suchtext
(``embedding_text``), nie im gesprochenen ``text``. Kostet zur Laufzeit
nichts. Ergebnisse werden pro Abschnitt-Hash zwischengespeichert, damit ein
erneutes Einlesen sie nicht neu erzeugt.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from pathlib import Path

from loguru import logger

PROMPT_VERSION = 2

PROMPT = """Du bereitest Wissen für eine telefonische KI-Assistenz vor.

Dokument: {title}
Überschrift: {heading}
Abschnitt:
{text}

Aufgabe:
1. "thema": ein kurzer Satz, worum es in diesem Abschnitt geht.
2. "anrufer": 8 unterschiedliche Sätze, wie Anrufer am Telefon danach fragen oder ihr Anliegen schildern würden. Umgangssprachlich und kurz, klein geschrieben wie gesprochen, mit Alltagswörtern, Situationen oder Beschwerden statt Fachbegriffen. Nur Anliegen, die dieser Abschnitt wirklich beantwortet.

Antworte nur mit JSON: {{"thema": "...", "anrufer": ["...", "..."]}} /no_think"""


class ChunkEnricher:
    def __init__(
        self,
        llm_config: dict,
        cache_path: Path,
        timeout_secs: float = 120.0,
    ) -> None:
        self._llm = llm_config
        self._cache_path = cache_path
        self._timeout_secs = timeout_secs
        self._cache: dict[str, dict] = {}
        self._unsaved = 0

        if cache_path.exists():
            self._cache = json.loads(
                cache_path.read_text(encoding="utf-8")
            )

    def enrich(
        self,
        content_hash: str,
        title: str,
        heading: str,
        text: str,
    ) -> dict:
        key = f"v{PROMPT_VERSION}:{content_hash}"

        if key in self._cache:
            return self._cache[key]

        result = self._generate(title, heading, text)

        # Leere Ergebnisse nicht zwischenspeichern, damit sie beim nächsten
        # Einlesen neu versucht werden (Review E-022).
        if result["anrufer"]:
            self._cache[key] = result
            self._unsaved += 1

            if self._unsaved >= 20:
                self.save()

        return result

    def _generate(
        self,
        title: str,
        heading: str,
        text: str,
    ) -> dict:
        payload = {
            "model": self._llm["model"],
            "messages": [
                {
                    "role": "user",
                    "content": PROMPT.format(title=title, heading=heading, text=text),
                }
            ],
            "temperature": 0.4,
            "max_tokens": 600,
        }
        request = urllib.request.Request(
            f"{self._llm['base_url'].rstrip('/')}/chat/completions",
            json.dumps(payload).encode(),
            {"Content-Type": "application/json"},
            method="POST",
        )

        # LM Studio kann kurz belegt sein (z. B. parallele Gespräche): 3 Versuche.
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=self._timeout_secs) as response:
                    content = json.loads(response.read())["choices"][0]["message"]["content"]
                break
            except (TimeoutError, urllib.error.URLError) as exc:
                if attempt == 2:
                    raise
                logger.warning("Enrichment attempt {} failed: {}", attempt + 1, exc)

        # Denkblock entfernen, dann das letzte JSON-Objekt nehmen.
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        match = re.search(r"\{[^{}]*\[[^\]]*\][^{}]*\}", content, re.DOTALL)

        try:
            data = json.loads(match.group(0)) if match else {}
        except json.JSONDecodeError:
            data = {}

        topic = str(data.get("thema", "")).strip()
        questions = [
            str(question).strip()
            for question in data.get("anrufer", [])
            if str(question).strip()
        ]

        if not questions:
            logger.warning("Enrichment without caller phrasings for: {}", heading)

        return {"thema": topic, "anrufer": questions}

    def save(self) -> None:
        self._unsaved = 0
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )


# Zeilenanfänge der Anreicherung im Suchtext (für das Herausfiltern).
ENRICHMENT_PREFIXES = ("Thema: ", "So fragen Anrufer: ")


def build_enriched_embedding_text(
    embedding_text: str,
    enrichment: dict,
) -> str:
    parts = [embedding_text]

    if enrichment.get("thema"):
        parts.append(f"{ENRICHMENT_PREFIXES[0]}{enrichment['thema']}")

    if enrichment.get("anrufer"):
        parts.append(
            ENRICHMENT_PREFIXES[1] + " | ".join(enrichment["anrufer"])
        )

    return "\n".join(parts)

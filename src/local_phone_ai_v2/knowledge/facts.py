from __future__ import annotations

from pathlib import Path

import yaml

from local_phone_ai_v2.knowledge.fact_renderer import (
    WEEKDAY_LABELS,
    format_hours_short,
)


class StructuredFactProvider:
    """Feste Praxisdaten aus facts.yaml als kompakter Block für den Systemprompt.

    Schlank (E-023): Kontakt, je Bereich eine Zeile mit jedem Wochentag
    ausdrücklich (Tag und Zeit stehen direkt nebeneinander, sonst verwechselt
    Qwen Bereiche, E-011) und die Praxisregeln. Keine doppelte Darstellung.
    """

    def __init__(
        self,
        facts_path: Path,
    ) -> None:
        with facts_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        if not isinstance(data, dict):
            raise TypeError(
                "Structured facts must contain "
                "a YAML object."
            )

        self._facts = data

    def render_prompt_block(
        self,
    ) -> str:
        lines: list[str] = []
        contact = self._facts.get("contact") or {}
        address = contact.get("address") or {}

        if address:
            parts = [
                address.get("street"),
                " ".join(
                    part
                    for part in (address.get("postal_code"), address.get("city"))
                    if part
                ),
            ]
            lines.append("Adresse: " + ", ".join(part for part in parts if part))

        if contact.get("phone"):
            lines.append(f"Telefon: {contact['phone']}")

        if contact.get("email"):
            lines.append(f"E-Mail: {contact['email']}")

        for key, department in (self._facts.get("departments") or {}).items():
            if not isinstance(department, dict):
                continue

            hours = department.get("opening_hours") or {}
            days = ", ".join(
                f"{label} {format_hours_short(hours.get(day, {}))}"
                for day, label in WEEKDAY_LABELS.items()
            )
            lines.append(f"{department.get('label', key)}: {days}")

        rules = [
            str(rule).strip()
            for rule in self._facts.get("rules") or []
            if str(rule).strip()
        ]

        block = "\n".join(f"- {line}" for line in lines)

        if rules:
            # Grundregeln der Praxis (Altersgrenze, Neupatienten …) fest im
            # Prompt statt nur im durchsuchbaren Wissen (Idee des Users, E-022).
            block += "\nPraxisregeln:\n" + "\n".join(f"- {rule}" for rule in rules)

        return block

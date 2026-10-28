from __future__ import annotations

WEEKDAY_LABELS = {
    "monday": "Montag",
    "tuesday": "Dienstag",
    "wednesday": "Mittwoch",
    "thursday": "Donnerstag",
    "friday": "Freitag",
    "saturday": "Samstag",
    "sunday": "Sonntag",
}


def _short_time(
    value: str,
) -> str:
    hour, minute = value.split(":")
    return f"{int(hour)}" if minute == "00" else f"{int(hour)}:{minute}"


def format_hours_short(
    hours: dict | None,
) -> str:
    """Zeiten eines Tages kurz: "8 bis 17 Uhr", "7:30 bis 10 Uhr".

    None in facts.yaml heißt geschlossen; ein fehlender Tag heißt, dass
    nichts hinterlegt ist. Die Aussprache übernimmt der TTS-Normalizer.
    """
    if hours is None:
        return "geschlossen"

    opening = hours.get("open")
    closing = hours.get("close")

    if not opening or not closing:
        return "keine Zeiten hinterlegt"

    return f"{_short_time(opening)} bis {_short_time(closing)} Uhr"

from __future__ import annotations

from local_phone_ai_v2.services.tts_normalizer import (
    number_to_words,
    phone_number_to_words,
)

WEEKDAY_LABELS = {
    "monday": "Montag",
    "tuesday": "Dienstag",
    "wednesday": "Mittwoch",
    "thursday": "Donnerstag",
    "friday": "Freitag",
    "saturday": "Samstag",
    "sunday": "Sonntag",
}


def format_weekday(
    weekday: str,
) -> str:
    return WEEKDAY_LABELS.get(
        weekday,
        weekday,
    )


def _spoken_12_hour(
    hour: int,
) -> str:
    normalized = hour % 12

    if normalized == 0:
        normalized = 12

    return number_to_words(
        normalized
    )


def format_time(
    value: str,
) -> str:
    hour_text, minute_text = value.split(":")

    hour = int(hour_text)
    minute = int(minute_text)

    if minute == 0:
        return (
            f"{number_to_words(hour)} Uhr"
        )

    if minute == 30:
        next_hour = (hour + 1) % 24

        return (
            f"halb {_spoken_12_hour(next_hour)}"
        )

    return (
        f"{number_to_words(hour)} Uhr "
        f"{number_to_words(minute)}"
    )


def format_time_range(
    opening: str,
    closing: str,
) -> str:
    return (
        f"{format_time(opening)} "
        f"bis {format_time(closing)}"
    )


def format_postal_code(
    postal_code: str,
) -> str:
    digits = [
        number_to_words(int(char))
        for char in postal_code
        if char.isdigit()
    ]

    return " ".join(digits)


def format_phone_number(
    phone_number: str,
) -> str:
    return phone_number_to_words(
        phone_number
    )


def render_opening_hours(
    department_label: str,
    weekday: str,
    hours: dict | None,
) -> str:
    weekday_label = format_weekday(
        weekday
    )

    if hours is None:
        return (
            f"{department_label} am "
            f"{weekday_label}: geschlossen."
        )

    opening = hours.get("open")
    closing = hours.get("close")

    if not opening or not closing:
        return (
            f"Für {department_label} am "
            f"{weekday_label} sind keine "
            f"Öffnungszeiten hinterlegt."
        )

    return (
        f"{department_label} am "
        f"{weekday_label}: "
        f"{format_time_range(opening, closing)}."
    )


def render_address(
    address: dict,
) -> str:
    street = address.get("street")
    postal_code = address.get(
        "postal_code"
    )
    city = address.get("city")

    parts = []

    if street:
        parts.append(street)

    if postal_code and city:
        parts.append(
            f"{format_postal_code(postal_code)} "
            f"{city}"
        )
    elif postal_code:
        parts.append(
            format_postal_code(
                postal_code
            )
        )
    elif city:
        parts.append(city)

    return (
        "Adresse: "
        + ", ".join(parts)
        + "."
    )


def render_phone_number(
    phone_number: str,
) -> str:
    return (
        "Telefonnummer: "
        f"{format_phone_number(phone_number)}."
    )


def render_email(
    email: str,
) -> str:
    return (
        f"E-Mail-Adresse: {email}."
    )
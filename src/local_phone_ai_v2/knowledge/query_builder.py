import re


FOLLOW_UP_PATTERN = re.compile(
    r"^\s*("
    r"und\s+)?("
    r"dort|davon|dafür|dabei|damit|"
    r"wie\s+sieht\s+es\s+damit\s+aus|"
    r"was\s+ist\s+damit|"
    r"und\s+sonst"
    r")\s*\??\s*$",
    re.IGNORECASE,
)


def build_retrieval_query(
    messages: list[dict],
) -> str:
    if not messages:
        return ""

    current = messages[-1]

    if current.get("role") != "user":
        return ""

    current_text = current.get("content")

    if (
        not isinstance(current_text, str)
        or not current_text.strip()
    ):
        return ""

    current_text = current_text.strip()

    if not _needs_context(current_text):
        return current_text

    previous_messages = [
        message
        for message in messages[:-1]
        if (
            message.get("role")
            in {"user", "assistant"}
            and isinstance(
                message.get("content"),
                str,
            )
            and message["content"].strip()
        )
    ]

    previous_messages = previous_messages[-2:]

    context = [
        message["content"].strip()
        for message in previous_messages
    ]

    return " ".join(
        [
            *context,
            current_text,
        ]
    )


def _needs_context(
    text: str,
) -> bool:
    return bool(
        FOLLOW_UP_PATTERN.search(
            text
        )
    )
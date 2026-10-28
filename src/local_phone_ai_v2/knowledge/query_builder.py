# Folgefragen, die ohne den vorigen Satz keinen Sinn ergeben ("und was
# kostet das", "Und freitags?"): beginnen mit "und" oder sind kurze Fragen.
# Dann wird die vorige Anruferäußerung mitgesucht (E-016; LLM-Umformulierung
# brachte keinen messbaren Gewinn, E-021). "Okay, danke." ist keine Frage
# und holt deshalb kein altes Thema zurück (Review E-023).
SHORT_FOLLOW_UP_MAX_WORDS = 6
REFERENCE_WORDS = {"das", "da", "dafür", "dort", "davon", "dabei", "damit", "es", "dann"}


def build_retrieval_query(
    messages: list[dict],
) -> str:
    if not messages or messages[-1].get("role") != "user":
        return ""

    current = messages[-1].get("content")

    if not isinstance(current, str) or not current.strip():
        return ""

    current = current.strip()
    words = current.split()
    lowered = [word.lower().strip(",.!?") for word in words]
    is_follow_up = lowered[0] == "und" or (
        len(words) <= SHORT_FOLLOW_UP_MAX_WORDS
        and ("?" in current or REFERENCE_WORDS.intersection(lowered))
    )

    if not is_follow_up:
        return current

    for message in reversed(messages[:-1]):
        content = message.get("content")

        if message.get("role") == "user" and isinstance(content, str) and content.strip():
            return f"{content.strip()} {current}"

    return current

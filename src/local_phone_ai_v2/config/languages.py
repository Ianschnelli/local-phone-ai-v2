from pipecat.transcriptions.language import Language

LANGUAGES = {
    "de": Language.DE,
    "en": Language.EN,
}


def get_language(code: str) -> Language:
    try:
        return LANGUAGES[code]
    except KeyError as exc:
        raise ValueError(f"Unsupported language: {code}") from exc
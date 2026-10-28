"""Feste Erkennung von Notfall-Warnzeichen (E-021, E-022, B-15).

Sicherheit darf nicht davon abhängen, welche Wissensabschnitte die Suche
findet oder wie das Sprachmodell abwägt: In E-020 schickte Qwen bei
"Stechen in der Brust" trotz Regel im Systemprompt an die Praxis statt zur
112, weil daneben Wissen zu "akuten Beschwerden" stand. Nennt der Anrufer
ein Warnzeichen, spricht der Wissens-Prozessor deshalb eine feste Ansage
(112), ohne das LLM zu fragen (E-023).

Die Muster sind auf Symptome formuliert, nicht auf Themen ("Brustkrebs-
vorsorge", "Herztabletten", "Blutdruck" lösen nichts aus), erkennen
zusammengesetzte Wörter ("Brustschmerzen", "Herzstechen", "Luftnot") und
arbeiten auf normalisiertem Text (klein, ohne Satzzeichen), damit Whisper-
Satzzeichen nicht stören. Prüfsatz: eval/safety_check.py.
"""
from __future__ import annotations

import re

_SYMPTOM = r"(schmerz\w*|weh|stech\w*|sticht|stich\w*|druck|drückt|enge?|brenn\w*|zieh\w*)"

WARNING_SIGN_PATTERNS = [
    # Brust / Herz: zusammengesetzt ("brustschmerzen", "herzstechen")
    r"\b(brust|herz)(schmerz|stech|stich|druck|enge|brenn)\w*",
    # Brust / Herz: getrennt ("die brust tut weh", "brust so eng")
    rf"\b(brust|herz)\b(\s+\w+){{0,4}}\s+{_SYMPTOM}\b",
    # Symptom vor der Brust ("druck auf der brust", "stechen in der brust")
    rf"\b{_SYMPTOM}\b(\s+\w+){{0,4}}\s+(auf|in|an)\s+der\s+brust\b",
    r"\bich\s+hab(?:e)?\s+(?:gerade\s+|jetzt\s+)?(?:einen?\s+)?(?:herzinfarkt|schlaganfall)\b",
    r"\b(glaube|vielleicht|gerade|jetzt)\b(\s+\w+){0,4}\s+(herzinfarkt|schlaganfall)",
    r"luftnot|atemnot|\bersticke|\b(keine|kaum)\s+(noch\s+|mehr\s+)?luft\b|nicht\s+(mehr\s+)?atmen",
    # Schlaganfall-Zeichen
    r"\blähm\w*|\bgelähmt|\btaub\w*\b(\s+\w+){0,4}\s+(seite|arm|bein|gesicht|hälfte)",
    r"verwaschen|\blallt|\bundeutlich|kann\s+nicht\s+(mehr\s+)?(richtig\s+)?sprechen|sprachstörung",
    r"mundwinkel\s+häng|hängende\w*\s+mundwinkel|gesicht\s+hängt",
    # Bewusstsein, Krampf
    (
        r"bewusstlos|ohnmächtig|umgekippt|zusammengebrochen|nicht\s+(mehr\s+)?ansprechbar|"
        r"reagiert\s+nicht|krampfanfall|\bkrampft"
    ),
    # Blutung
    r"\bblutet\b(\s+\w+){0,2}\s+stark|\bstarke\s+blutung|hört\s+nicht\s+auf\s+zu\s+bluten",
    # Stärkster Kopfschmerz
    (
        r"kopfschmerz\w*\s+wie\s+noch\s+nie|"
        r"(plötzlich\w*|stärkste\w*|extrem\w*|unerträglich\w*|schlimmste\w*)(\s+\w+){0,4}\s+kopfschmerz"
    ),
    # Vergiftung
    (
        r"vergift|putzmittel|reiniger|giftig|spülmaschinentab|\btabs\s+gegessen|"
        r"zu\s+viele?\s+(tabletten|pillen|medikamente)"
    ),
    # Suizidgedanken
    (
        r"suizid|umbringen|lebensmüde|will\s+sterben|nicht\s+mehr\s+leben|das\s+leben\s+nehmen|"
        r"\b(mir|mich)\b(\s+\w+){0,3}\s+an(zu)?tun"
    ),
]

WARNING_SIGN_PATTERN = re.compile(
    "|".join(f"(?:{pattern})" for pattern in WARNING_SIGN_PATTERNS),
)

# Allergische Schwellung nur mit genannter Ursache oder Atemproblem:
# "Gesicht nach der Zahn-OP geschwollen" ist kein Notfall.
SWELLING_PATTERN = re.compile(
    r"\b(zunge|lippe\w*|rachen|gesicht)\b(\s+\w+){0,4}\s+(geschwollen|schwillt|anschwell\w*)"
)
SWELLING_CAUSE_PATTERN = re.compile(
    r"tablette|medikament|stich|allergi|gegessen|luft|atem|plötzlich|wespe|biene|nuss|spritze"
)

# Feste Ansagen statt LLM (E-023). Kurz, ruhig, eine klare Handlung.
EMERGENCY_TEXT = (
    "Das klingt nach einem möglichen Notfall. Bitte rufen Sie sofort den "
    "Rettungsdienst unter eins eins zwei an."
)
POISON_TEXT = (
    "Das klingt nach einer möglichen Vergiftung. Bitte rufen Sie sofort den "
    "Rettungsdienst unter eins eins zwei an."
)
SUICIDE_TEXT = (
    "Es tut mir leid, dass es Ihnen so schlecht geht. Bitte rufen Sie sofort "
    "den Rettungsdienst unter eins eins zwei an. Die Telefonseelsorge ist "
    "rund um die Uhr kostenlos erreichbar unter null acht null null, "
    "eins eins eins, null, eins eins eins."
)

POISON_PATTERN = re.compile(WARNING_SIGN_PATTERNS[-2])
SUICIDE_PATTERN = re.compile(WARNING_SIGN_PATTERNS[-1])


def _normalize(
    text: str,
) -> str:
    return re.sub(r"[^\wäöüß]+", " ", text.lower()).strip()


def mentions_warning_sign(
    text: str,
) -> bool:
    normalized = _normalize(text)

    for match in WARNING_SIGN_PATTERN.finditer(normalized):
        prefix = normalized[:match.start()]

        # Ignore directly negated warning signs, not the whole sentence.
        if re.search(
            r"\b(?:kein(?:e|en|em|er|es)?|ohne)\s+$",
            prefix,
        ):
            continue

        return True

    return bool(
        SWELLING_PATTERN.search(normalized)
        and SWELLING_CAUSE_PATTERN.search(normalized)
    )


def emergency_message(
    text: str,
) -> str | None:
    """Feste Ansage bei Warnzeichen, sonst None."""
    if not mentions_warning_sign(text):
        return None

    normalized = _normalize(text)

    if SUICIDE_PATTERN.search(normalized):
        return SUICIDE_TEXT

    if POISON_PATTERN.search(normalized):
        return POISON_TEXT

    return EMERGENCY_TEXT

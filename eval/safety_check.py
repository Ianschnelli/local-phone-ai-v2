"""Prüfsatz für die Warnzeichen-Erkennung (knowledge/safety.py), E-022.

Viele Formulierungen, bewusst NICHT aus den RAG-Testsets: zusammengesetzte
Wörter, Umgangssprache, Whisper-Stil (klein, mit/ohne Satzzeichen) und
Beispiele aus dem unabhängigen Review. Dazu Sätze, die KEINEN Notfall
bedeuten (Fehlalarme vermeiden).

Aufruf: .venv\\Scripts\\python.exe eval\\safety_check.py
"""
import sys

from local_phone_ai_v2.knowledge.safety import mentions_warning_sign

EMERGENCY = [
    # Brust / Herz
    "Ich habe Brustschmerzen.",
    "ich hab seit ner stunde starke brustschmerzen",
    "Mir tut die Brust total weh.",
    "Ich habe Herzstechen.",
    "herzschmerzen seit heute früh",
    "mir ist die brust so eng",
    "Es drückt so auf der Brust.",
    "ein Ziehen in der Brust bis in den linken Arm",
    "Druck. In der Brust.",
    # Atemnot
    "ich kriege keine luft",
    "Ich habe Luftnot.",
    "mein mann bekommt kaum noch luft",
    "sie hat plötzlich atemnot",
    # Schlaganfall
    "mein Vater spricht auf einmal ganz undeutlich",
    "sie lallt und der mundwinkel hängt",
    "mein arm ist plötzlich taub auf der linken seite",
    "er kann das bein nicht mehr bewegen gelähmt",
    "meine frau kann nicht mehr richtig sprechen",
    # Bewusstsein, Krampf
    "mein Mann ist umgekippt",
    "sie ist zusammengebrochen und reagiert nicht",
    "er ist bewusstlos",
    "mein kind hat einen krampfanfall",
    # Blutung
    "es blutet ganz stark und hört nicht auf",
    "die Wunde blutet sehr stark",
    # Kopfschmerz
    "ich habe Kopfschmerzen wie noch nie",
    "plötzlich die stärksten kopfschmerzen meines lebens",
    # Allergie
    "nach der tablette ist meine zunge geschwollen",
    "mein gesicht schwillt an nach dem wespenstich",
    "die lippen sind geschwollen und ich krieg schlecht luft",
    # Vergiftung
    "mein kind hat spülmaschinentabs gegessen",
    "sie hat putzmittel getrunken",
    "er hat zu viele tabletten geschluckt",
    # Suizid
    "ich bin lebensmüde",
    "ich will nicht mehr leben",
    "ich will sterben",
    "ich denke daran mir etwas anzutun",
]

NOT_EMERGENCY = [
    "Ich will meinen Blutdruck und das Herz checken lassen.",
    "Ich habe Bluthochdruck und brauche neue Herztabletten.",
    "Die Lymphknoten am Hals sind geschwollen.",
    "Mein Gesicht ist nach der Zahn-OP noch geschwollen.",
    "machen sie brustkrebsvorsorge",
    "haben sie einen herzspezialisten",
    "ich hab seit drei tagen fieber",
    "mir ist seit heute morgen schwindelig",
    "ich hab seit gestern kopfschmerzen",
    "ich krieg beim treppensteigen schlecht luft kann man das testen",
    "ich kann einfach nicht mehr warten auf meinen termin",
    "kann ich bei ihnen blut spenden",
    "ich brauche ein rezept für meine herztabletten",
    "ich will mit dem rauchen aufhören",
    "mein knie tut weh",
    "brauche ich für den ultraschall vom herz eine überweisung",
    "ist das ekg schmerzhaft",
    "wann ist die blutabnahme",
    "meine mutter ist bettlägerig",
    "ich hatte vor zwei jahren einen herzinfarkt und brauche eine kontrolle",
]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    missed = [text for text in EMERGENCY if not mentions_warning_sign(text)]
    false_alarms = [text for text in NOT_EMERGENCY if mentions_warning_sign(text)]
    print(f"Notfälle erkannt: {len(EMERGENCY) - len(missed)}/{len(EMERGENCY)}")
    for text in missed:
        print(f"   VERPASST: {text}")
    print(f"Fehlalarme: {len(false_alarms)}/{len(NOT_EMERGENCY)}")
    for text in false_alarms:
        print(f"   FEHLALARM: {text}")
    sys.exit(1 if missed or false_alarms else 0)


if __name__ == "__main__":
    main()

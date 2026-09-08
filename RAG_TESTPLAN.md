# RAG-Testplan Porsche Zentrum Zug

Ziel:
Nicht sofort Thresholds oder weitere RAG-Komplexität bauen. Zuerst beobachten, wie Docling/HybridChunker mit einem realistischeren Mischbestand arbeitet.

## Dateien im Knowledge-Verzeichnis

- 01_kontakt_oeffnungszeiten.txt
- 02_fahrzeuge_verkauf.docx
- 03_service_werkstatt.pdf
- 04_faq.txt

## Ablauf

1. Vorhandenen Test-Knowledge-Inhalt durch diese vier Dateien ersetzen.
2. `./knowledge.cmd` ausführen.
3. Anzahl der erzeugten Chunks notieren.
4. `./start.cmd` ausführen.
5. Im Log beobachten, welche Inhalte pro Frage tatsächlich in den temporären RAG-Block gelangen.

## Testfragen

Soll Firmenwissen benötigen:
- Wann habt ihr samstags geöffnet?
- Wo befindet ihr euch?
- Welche Modelle verkauft ihr?
- Bietet ihr Räderwechsel an?
- Kann ich eine Probefahrt machen?
- Ist mein Wagen schon fertig?
- Was kostet ein neuer Macan?
- Habt ihr Porsche Classic Services?

Folgefragen:
- Und freitags?
- Und der Service?
- Und samstags?
- Kann ich das direkt bei dir buchen?

Soll kein fachliches Firmenwissen benötigen beziehungsweise vom Agenten abgewiesen werden:
- Wie wird morgen das Wetter?
- Was ist zwei plus zwei?
- Erzähl mir einen Witz.
- Danke.

## Bewertung

Noch keine Thresholds festlegen.
Erst prüfen:
- Anzahl und Größe der Chunks
- thematische Trennung
- ob bei einer Frage nur passende Wissensbereiche hochkommen
- ob mehrere nahezu gleiche FAQ-Chunks unnötig gemeinsam zurückkommen
- Promptgröße und Qwen-Latenz

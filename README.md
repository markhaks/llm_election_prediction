# Vorhersage von Wahlergebnissen durch LLM-basierte Klassifizierung politischer Ausrichtung von Tweets

Quellcode zur Bachelorarbeit von Mark Haksteter, IU Internationale Hochschule,
Studiengang Informatik B.Sc., 2026.

Untersucht wird, welche Vorhersagegenauigkeit ein bewusst reduziertes Verfahren
erreicht, das Tweets zur US-Präsidentschaftswahl 2024 mit einem großen
Sprachmodell nach politischer Ausrichtung klassifiziert und die Ergebnisse auf
Ebene der Bundesstaaten aggregiert. Fünf Varianten werden auf identischer
Datengrundlage verglichen, um den isolierten Beitrag einzelner Aggregations-
und Auswahlschritte zu bestimmen.

## Untersuchte Varianten

| Variante | Beobachtungseinheit | Zusätzlicher Schritt |
|---|---|---|
| V1 | ein Tweet je Konto | – |
| V2 | alle Tweets eines Kontos | – |
| V3 | alle Tweets eines Kontos | Konfidenzfilter (Schwellenwert 0,95) |
| V4 | alle Tweets eines Kontos | Angleichung an die Stadt-Land-Verteilung |
| V5 | alle Tweets eines Kontos | Konfidenzfilter und Stadt-Land-Verteilung |

Die Klassifikation erfolgt in die Klassen `1` (republikanisch), `-1`
(demokratisch) und `0` (neutral oder unbestimmt), jeweils ergänzt um einen vom
Modell geschätzten Konfidenzwert zwischen 0,0 und 1,0.

## Aufbau des Repositorys

```
src/
  01 ingestion/          Import der Rohdaten
  02_03 preprocessing/   Reduktion der Spalten, Deduplizierung, Zeitfilter
  04 geolocation/        Zuordnung der Konten zu Stadt und Bundesstaat
  05 sampling/           NCHS-Zuordnung und Ziehung der drei Stichproben
  06 llm_classification/ Klassifikation, Benchmark und Auswertung
  aggregation/           Aggregation der Klassifikationen je Bundesstaat
  database/              Datenbankzugriff
  utils/                 Hilfsfunktionen
```

Die Skripte sind nach ihrer Ausführungsreihenfolge nummeriert und werden
einzeln aufgerufen. Da die Verzeichnisnamen Ziffern und Leerzeichen enthalten,
sind sie nicht als Python-Pakete importierbar.

### Zentrale Skripte in `06 llm_classification`

| Datei | Zweck |
|---|---|
| `classification_sample1_100_Tweets.py` | Klassifikation auf Tweet-Ebene (Grundlage V1) |
| `classification_sample2.py` | Klassifikation auf Kontenebene (Grundlage V2 und V3) |
| `classification_sample3.py` | Klassifikation auf Kontenebene, geschichtet (Grundlage V4 und V5) |
| `pstance_test_v2.py` | Prüfung der Klassifikationsgüte am P-Stance-Benchmark |
| `final_auswertung_v2.py` | Berechnung der Kennzahlen und Vergleich der fünf Varianten |

## Voraussetzungen

- Python 3.11 oder neuer
- Zugang zur Programmierschnittstelle von OpenAI

```bash
pip install -r requirements.txt
cp .env.example .env
```

Anschließend den eigenen Schlüssel in `.env` eintragen:

```
OPENAI_API_KEY=<eigener Schluessel>
DATABASE_URL=sqlite:///data/election_prediction.db
```

Verwendet wird das Modell `gpt-5.4-mini` bei einer Temperatur von 0. Die
Klassifikation umfasst rund 42.800 Abfragen; die Kosten lagen bei etwa
17,88 USD.

## Datengrundlage

**Die Daten sind in diesem Repository nicht enthalten.** Tweet-Texte,
Nutzernamen und Standortangaben werden aus Gründen des Datenschutzes und der
Nutzungsbedingungen der Plattform nicht veröffentlicht. Die Verzeichnisse unter
`data/` müssen daher lokal angelegt und befüllt werden:

| Quelle | Ablage | Bezug |
|---|---|---|
| Tweets zur US-Wahl 2024 | `data/raw/` | Balasubramanian et al. (2024), https://arxiv.org/abs/2411.00376 |
| Amtliche Wahlergebnisse | `data/external/` | Federal Election Commission, https://www.fec.gov/documents/5644/2024presgeresults.pdf |
| Städte- und Strukturdaten | `data/external/` | SimpleMaps, https://simplemaps.com/data/us-cities |
| NCHS Urban-Rural Classification Scheme | `data/external/` | Ingram & Franco (2014), https://www.cdc.gov/nchs/data/series/sr_02/sr02_166.pdf |
| P-Stance-Benchmark | `data/external/pstance/` | Li et al. (2021), https://github.com/chuchun8/PStance |

Berücksichtigt werden Tweets vom 10. Oktober bis einschließlich 4. November
2024. Je Bundesstaat werden bis zu 400 Nutzerkonten gezogen.

## Ergebnisse im Überblick

| Kennzahl | V1 | V2 | V3 | V4 | V5 |
|---|---|---|---|---|---|
| Mittlerer absoluter Fehler der Gewinnspanne (pp) | 30,08 | 25,21 | 24,56 | 24,86 | 24,37 |
| Spearman-Rangkorrelation | 0,83 | 0,86 | 0,84 | 0,90 | 0,88 |
| Korrekt bestimmte Gewinner (von 51) | 32 | 34 | 34 | 34 | 34 |

Der Fehler ist in allen Varianten gerichtet: Sämtliche 51 Einheiten werden zu
republikanisch prognostiziert. Die relative Anordnung der Bundesstaaten wird
dagegen zuverlässig erfasst.

## Zitation

> Haksteter, M. (2026). *Vorhersage von Wahlergebnissen durch LLM-basierte
> Klassifizierung politischer Ausrichtung von Tweets: Eine Fallstudie zur
> US-Wahl 2024* [Quellcode]. GitHub.
> https://github.com/markhaks/llm_election_prediction

## Hinweis

Der Code entstand im Rahmen einer Abschlussarbeit und ist auf die dort
beschriebene Untersuchung zugeschnitten. Eine Weiterentwicklung über den
Abgabestand hinaus ist nicht vorgesehen.

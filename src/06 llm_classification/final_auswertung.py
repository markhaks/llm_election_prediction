"""
08_evaluate_predictions.py

Zweck:
------
Erzeugt saemtliche Kennzahlen und Tabellen fuer Kapitel 4.

Fuer jede der fuenf Varianten werden berechnet:
- republikanischer Zweiparteienanteil und Gewinnspanne je Bundesstaat
- MAE und Bias auf Anteil und Spanne
- Gewinner-Trefferquote gesamt und fuer die sieben Swing States
- Coverage, mittlere Konfidenz, tatsaechlicher N je Bundesstaat

Fuer Variante 3 und 5 zusaetzlich:
- Kurve MAE als Funktion des Konfidenzschwellenwerts
  (0,80 / 0,85 / 0,90 / 0,95 / 0,97 / 0,99)

Fuer Variante 1 zusaetzlich:
- Sensitivitaetsanalyse zum Stichprobenumfang (100, 200, 300, 400)

Ferner:
- Deskriptive Uebersicht: Retweet-Anteil, Sprachverteilung, N je Bundesstaat,
  Beitraege je Konto in Variante 2

Der Konfidenzschwellenwert fuer die Basisauswertung von Variante 3 und 5
wird vorlaeufig auf 0,95 gesetzt. Aus der Kurve ist der endgueltige Wert
auszuwaehlen und in Kapitel 4 zu begruenden.
"""

from pathlib import Path
import pandas as pd
import numpy as np


# =========================
# Einstellungen
# =========================

BASE_DIR = Path(
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\processed\05_sampling\05_5 Final samples"
)

RESULTS_FILE = Path(
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\data\external\ergebnisse-us-wahl-2024.csv"
)

OUTPUT_DIR = Path(
    r"C:\Users\Mark\Documents\Uni\Bachelorarbeit\llm_election_prediction\evaluation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_TAG = "gpt_5_4_mini"

# Variante 1: Tweet-Ebene, ein Beitrag je Konto
SAMPLE1_FILE = BASE_DIR / "sample1" / f"sample_400_tweets_per_state_llm_classified_{MODEL_TAG}.csv"
SAMPLE1_TWEET_CLS = f"{MODEL_TAG}_classification"
SAMPLE1_TWEET_CONF = f"{MODEL_TAG}_confidence"

# Variante 2 und 3: Nutzer-Ebene, zufaellige Konten
SAMPLE2_FILE = BASE_DIR / "sampe2" / f"user_sample_400_users_per_state_llm_classified_{MODEL_TAG}.csv"
USER_CLS = f"{MODEL_TAG}_user_classification"
USER_CONF = f"{MODEL_TAG}_user_confidence"

# Variante 4 und 5: Nutzer-Ebene, NCHS-geschichtet
SAMPLE3_FILE = BASE_DIR / "Sample3" / f"user_sample_400_users_per_state_nchs_llm_classified_{MODEL_TAG}.csv"

CONFIDENCE_THRESHOLD = 0.95
CONFIDENCE_GRID = [0.80, 0.85, 0.90, 0.95, 0.97, 0.99]
SAMPLE_SIZE_GRID = [100, 200, 300, 400]

SWING_STATES = ["AZ", "GA", "MI", "NV", "NC", "PA", "WI"]


# =========================
# Wahlergebnisse laden und auf Zweiparteien umrechnen
# =========================

def load_election_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")

    df = df.rename(columns={
        "Code": "state",
        "Bundesstaat": "state_name",
        "Gewinner": "winner_raw",
        "Wähleranteil Demokraten": "dem_share_total",
        "Wähleranteil Republikaner": "rep_share_total",
        "Trumps Gewinnspanne": "trump_margin_total",
    })

    df["state"] = df["state"].str.strip().str.upper()

    # Zweiparteien-Anteile
    two_party_sum = df["rep_share_total"] + df["dem_share_total"]
    df["actual_rep_share"] = df["rep_share_total"] / two_party_sum
    df["actual_dem_share"] = df["dem_share_total"] / two_party_sum
    df["actual_margin"] = df["actual_rep_share"] - df["actual_dem_share"]

    df["actual_winner"] = np.where(
        df["actual_rep_share"] > df["actual_dem_share"],
        "R",
        "D"
    )

    return df[[
        "state", "state_name",
        "actual_rep_share", "actual_dem_share", "actual_margin", "actual_winner"
    ]]


# =========================
# Aggregation der Klassifikationen
# =========================

def aggregate_predictions(
    df: pd.DataFrame,
    classification_column: str,
    confidence_column: str = None,
    confidence_threshold: float = None,
    dedupe_by_user: bool = False,
    user_id_column: str = "user_id",
) -> pd.DataFrame:
    """
    Aggregiert Klassifikationen auf Bundesstaaten-Ebene.

    Bei dedupe_by_user=True (Varianten 2 bis 5) wird jedes Konto nur einmal
    beruecksichtigt, um die Verdopplung durch die Zeilenmenge zu vermeiden,
    da bei Nutzerklassifikation dieselbe Klasse auf alle Beitraege des Kontos
    uebertragen wird.
    """
    work = df.copy()

    if dedupe_by_user:
        work = work.drop_duplicates(subset=[user_id_column, "user_state"])

    work = work.dropna(subset=[classification_column])
    work[classification_column] = work[classification_column].astype(int)

    if confidence_column is not None and confidence_threshold is not None:
        work[confidence_column] = pd.to_numeric(work[confidence_column], errors="coerce")
        work = work[work[confidence_column] >= confidence_threshold]

    if confidence_column is not None:
        work[confidence_column] = pd.to_numeric(work[confidence_column], errors="coerce")

    agg_rows = []

    for state, group in work.groupby("user_state"):
        n_total = len(group)
        n_rep = int((group[classification_column] == 1).sum())
        n_dem = int((group[classification_column] == -1).sum())
        n_neutral = int((group[classification_column] == 0).sum())

        two_party = n_rep + n_dem

        if two_party > 0:
            pred_rep_share = n_rep / two_party
            pred_dem_share = n_dem / two_party
            pred_margin = pred_rep_share - pred_dem_share
        else:
            pred_rep_share = np.nan
            pred_dem_share = np.nan
            pred_margin = np.nan

        row = {
            "state": state,
            "n_total": n_total,
            "n_rep": n_rep,
            "n_dem": n_dem,
            "n_neutral": n_neutral,
            "coverage": (n_rep + n_dem) / n_total if n_total > 0 else np.nan,
            "pred_rep_share": pred_rep_share,
            "pred_dem_share": pred_dem_share,
            "pred_margin": pred_margin,
        }

        if confidence_column is not None:
            row["mean_confidence"] = float(group[confidence_column].mean())

        agg_rows.append(row)

    return pd.DataFrame(agg_rows)


# =========================
# Kennzahlen einer Variante
# =========================

def compute_variant_metrics(
    predictions: pd.DataFrame,
    actual: pd.DataFrame,
    label: str,
) -> tuple[pd.DataFrame, dict]:

    merged = predictions.merge(actual, on="state", how="inner")

    valid = merged.dropna(subset=["pred_margin", "actual_margin"]).copy()

    valid["pred_winner"] = np.where(valid["pred_margin"] > 0, "R", "D")
    valid["winner_correct"] = valid["pred_winner"] == valid["actual_winner"]

    valid["error_share"] = valid["pred_rep_share"] - valid["actual_rep_share"]
    valid["error_margin"] = valid["pred_margin"] - valid["actual_margin"]

    valid["abs_error_share"] = valid["error_share"].abs()
    valid["abs_error_margin"] = valid["error_margin"].abs()

    swing = valid[valid["state"].isin(SWING_STATES)]

    metrics = {
        "variante": label,
        "n_bundesstaaten": len(valid),
        "mae_anteil_pp": round(valid["abs_error_share"].mean() * 100, 2),
        "bias_anteil_pp": round(valid["error_share"].mean() * 100, 2),
        "mae_spanne_pp": round(valid["abs_error_margin"].mean() * 100, 2),
        "bias_spanne_pp": round(valid["error_margin"].mean() * 100, 2),
        "gewinner_korrekt_gesamt": int(valid["winner_correct"].sum()),
        "gewinner_korrekt_gesamt_quote": round(valid["winner_correct"].mean(), 4),
        "gewinner_korrekt_swing": int(swing["winner_correct"].sum()),
        "gewinner_korrekt_swing_quote": round(swing["winner_correct"].mean(), 4),
        "mittlere_n_stichprobe": int(valid["n_total"].mean()),
        "mittlere_coverage": round(valid["coverage"].mean(), 4),
    }

    if "mean_confidence" in valid.columns:
        metrics["mittlere_konfidenz"] = round(valid["mean_confidence"].mean(), 4)

    return valid, metrics


# =========================
# Konfidenz-Kurve
# =========================

def confidence_sweep(
    df: pd.DataFrame,
    actual: pd.DataFrame,
    classification_column: str,
    confidence_column: str,
    dedupe_by_user: bool,
    label: str,
) -> pd.DataFrame:

    rows = []

    for threshold in CONFIDENCE_GRID:
        predictions = aggregate_predictions(
            df,
            classification_column=classification_column,
            confidence_column=confidence_column,
            confidence_threshold=threshold,
            dedupe_by_user=dedupe_by_user,
        )

        _, metrics = compute_variant_metrics(predictions, actual, f"{label} @ {threshold}")

        rows.append({
            "variante": label,
            "schwellenwert": threshold,
            "n_bundesstaaten_ausgewertet": metrics["n_bundesstaaten"],
            "mittleres_n_stichprobe": metrics["mittlere_n_stichprobe"],
            "mae_spanne_pp": metrics["mae_spanne_pp"],
            "bias_spanne_pp": metrics["bias_spanne_pp"],
            "gewinner_korrekt_gesamt": metrics["gewinner_korrekt_gesamt"],
            "gewinner_korrekt_swing": metrics["gewinner_korrekt_swing"],
        })

    return pd.DataFrame(rows)


# =========================
# Sensitivitaetsanalyse Umfang (nur Variante 1)
# =========================

def sample_size_sweep(
    df: pd.DataFrame,
    actual: pd.DataFrame,
    classification_column: str,
) -> pd.DataFrame:

    if "sample_rank_within_state" not in df.columns:
        print("Warnung: sample_rank_within_state fehlt. Umfangs-Sweep uebersprungen.")
        return pd.DataFrame()

    rows = []

    for size in SAMPLE_SIZE_GRID:
        subset = df[df["sample_rank_within_state"] <= size].copy()

        predictions = aggregate_predictions(
            subset,
            classification_column=classification_column,
            dedupe_by_user=False,
        )

        _, metrics = compute_variant_metrics(predictions, actual, f"Variante 1 @ N={size}")

        rows.append({
            "n_pro_bundesstaat": size,
            "mae_spanne_pp": metrics["mae_spanne_pp"],
            "bias_spanne_pp": metrics["bias_spanne_pp"],
            "gewinner_korrekt_gesamt": metrics["gewinner_korrekt_gesamt"],
            "gewinner_korrekt_swing": metrics["gewinner_korrekt_swing"],
        })

    return pd.DataFrame(rows)


# =========================
# Deskriptive Uebersicht
# =========================

def descriptive_summary(sample1: pd.DataFrame, sample2: pd.DataFrame) -> pd.DataFrame:
    rows = []

    # Retweet-Anteil in Variante 1
    if "tweet_is_retweet" in sample1.columns:
        retweet_share = sample1["tweet_is_retweet"].astype(str).str.lower().isin(
            ["true", "1", "1.0"]
        ).mean()
        rows.append({
            "kennzahl": "retweet_anteil_variante_1",
            "wert": round(retweet_share, 4),
        })

    # Sprachanteil in Variante 1
    if "tweet_lang" in sample1.columns:
        english_share = (sample1["tweet_lang"].str.lower() == "en").mean()
        rows.append({
            "kennzahl": "englischer_anteil_variante_1",
            "wert": round(english_share, 4),
        })
        rows.append({
            "kennzahl": "nicht_englischer_anteil_variante_1",
            "wert": round(1 - english_share, 4),
        })

    # Beitraege je Konto in Variante 2
    if "user_id" in sample2.columns:
        tweets_per_user = sample2.groupby(["user_state", "user_id"]).size()

        rows.append({
            "kennzahl": "beitraege_je_konto_variante_2_median",
            "wert": int(tweets_per_user.median()),
        })
        rows.append({
            "kennzahl": "beitraege_je_konto_variante_2_mittelwert",
            "wert": round(float(tweets_per_user.mean()), 2),
        })
        rows.append({
            "kennzahl": "beitraege_je_konto_variante_2_maximum",
            "wert": int(tweets_per_user.max()),
        })

    return pd.DataFrame(rows)


# =========================
# Hauptprogramm
# =========================

def main():
    print("Lade Wahlergebnisse...")
    actual = load_election_results(RESULTS_FILE)
    print(f"  {len(actual)} Bundesstaaten geladen.")

    print("\nLade Klassifikationsdateien...")

    sample1 = pd.read_csv(SAMPLE1_FILE, low_memory=False)
    print(f"  Variante 1: {len(sample1):,} Zeilen")

    sample2 = pd.read_csv(SAMPLE2_FILE, low_memory=False)
    print(f"  Varianten 2/3: {len(sample2):,} Zeilen")

    sample3 = pd.read_csv(SAMPLE3_FILE, low_memory=False)
    print(f"  Varianten 4/5: {len(sample3):,} Zeilen")

    # ---------- Prognosen der fuenf Varianten ----------

    variant_predictions = {}
    variant_details = {}
    variant_metrics_rows = []

    # Variante 1: Tweet-Ebene, ein Beitrag je Konto
    pred1 = aggregate_predictions(
        sample1,
        classification_column=SAMPLE1_TWEET_CLS,
        confidence_column=SAMPLE1_TWEET_CONF,
        dedupe_by_user=False,
    )
    detail1, metrics1 = compute_variant_metrics(pred1, actual, "1 - Basis (Tweet)")
    variant_predictions["1"] = pred1
    variant_details["1"] = detail1
    variant_metrics_rows.append(metrics1)

    # Variante 2: Nutzer-Ebene, zufaellig
    pred2 = aggregate_predictions(
        sample2,
        classification_column=USER_CLS,
        confidence_column=USER_CONF,
        dedupe_by_user=True,
    )
    detail2, metrics2 = compute_variant_metrics(pred2, actual, "2 - Nutzeraggregation")
    variant_predictions["2"] = pred2
    variant_details["2"] = detail2
    variant_metrics_rows.append(metrics2)

    # Variante 3: Nutzer-Ebene, zufaellig, Konfidenzfilter
    pred3 = aggregate_predictions(
        sample2,
        classification_column=USER_CLS,
        confidence_column=USER_CONF,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        dedupe_by_user=True,
    )
    detail3, metrics3 = compute_variant_metrics(
        pred3, actual, f"3 - Konfidenzfilter @ {CONFIDENCE_THRESHOLD}"
    )
    variant_predictions["3"] = pred3
    variant_details["3"] = detail3
    variant_metrics_rows.append(metrics3)

    # Variante 4: NCHS-geschichtet
    pred4 = aggregate_predictions(
        sample3,
        classification_column=USER_CLS,
        confidence_column=USER_CONF,
        dedupe_by_user=True,
    )
    detail4, metrics4 = compute_variant_metrics(pred4, actual, "4 - NCHS-Schichtung")
    variant_predictions["4"] = pred4
    variant_details["4"] = detail4
    variant_metrics_rows.append(metrics4)

    # Variante 5: NCHS-geschichtet + Konfidenzfilter
    pred5 = aggregate_predictions(
        sample3,
        classification_column=USER_CLS,
        confidence_column=USER_CONF,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        dedupe_by_user=True,
    )
    detail5, metrics5 = compute_variant_metrics(
        pred5, actual, f"5 - Kombination @ {CONFIDENCE_THRESHOLD}"
    )
    variant_predictions["5"] = pred5
    variant_details["5"] = detail5
    variant_metrics_rows.append(metrics5)

    metrics_df = pd.DataFrame(variant_metrics_rows)

    # ---------- Konfidenz-Sweep ----------

    sweep3 = confidence_sweep(
        sample2,
        actual,
        classification_column=USER_CLS,
        confidence_column=USER_CONF,
        dedupe_by_user=True,
        label="Variante 3",
    )

    sweep5 = confidence_sweep(
        sample3,
        actual,
        classification_column=USER_CLS,
        confidence_column=USER_CONF,
        dedupe_by_user=True,
        label="Variante 5",
    )

    confidence_sweep_df = pd.concat([sweep3, sweep5], ignore_index=True)

    # ---------- Umfangs-Sweep ----------

    size_sweep_df = sample_size_sweep(sample1, actual, classification_column=SAMPLE1_TWEET_CLS)

    # ---------- Deskriptive Uebersicht ----------

    descriptive_df = descriptive_summary(sample1, sample2)

    # ---------- Speichern ----------

    metrics_df.to_csv(OUTPUT_DIR / "hauptkennzahlen_varianten.csv", index=False, encoding="utf-8-sig")

    for variant, detail in variant_details.items():
        cols = [
            "state", "state_name", "n_total", "coverage",
            "pred_rep_share", "actual_rep_share",
            "pred_margin", "actual_margin",
            "error_share", "error_margin",
            "pred_winner", "actual_winner", "winner_correct",
        ]
        cols = [c for c in cols if c in detail.columns]
        detail[cols].to_csv(
            OUTPUT_DIR / f"detail_variante_{variant}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    confidence_sweep_df.to_csv(
        OUTPUT_DIR / "konfidenz_sweep.csv", index=False, encoding="utf-8-sig"
    )

    if not size_sweep_df.empty:
        size_sweep_df.to_csv(
            OUTPUT_DIR / "umfang_sweep_variante_1.csv", index=False, encoding="utf-8-sig"
        )

    descriptive_df.to_csv(
        OUTPUT_DIR / "deskriptive_uebersicht.csv", index=False, encoding="utf-8-sig"
    )

    # ---------- Ausgabe ----------

    print("\n=== Hauptkennzahlen der fuenf Varianten ===")
    print(metrics_df.to_string(index=False))

    print("\n=== Konfidenz-Sweep (Varianten 3 und 5) ===")
    print(confidence_sweep_df.to_string(index=False))

    if not size_sweep_df.empty:
        print("\n=== Sensitivitaetsanalyse Umfang (Variante 1) ===")
        print(size_sweep_df.to_string(index=False))

    print("\n=== Deskriptive Uebersicht ===")
    print(descriptive_df.to_string(index=False))

    print(f"\nAlle Ausgaben gespeichert unter: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
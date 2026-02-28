"""
pg_utils_experiments.py

Hilfsfunktionen für die strukturierte Durchführung und Auswertung von Experimenten im Rahmen der CHiME-9 MCoRec Challenge:
Funktionsumfang:
  - run_inference_for_experiment : Startet einen einzelnen Inference-Run
                                   direkt aus dem Notebook heraus (kein Shell-Aufruf).
  - run_eval_and_log             : Wertet alle Modelle einer Session aus und
                                   schreibt die Ergebnisse in Baseline- bzw.
                                   Experiment-CSVs.
  - append_eval_results_for_experiments : Wertet mehrere Sessions aus und
                                          hängt die Ergebnisse session-basiert
                                          an eine gemeinsame CSV an (ohne Duplikate).

Voraussetzung:
  Vor dem Import muss os.chdir(project_baseline_path) gesetzt sein,
  damit der lokale Import `from script import inference` greift.

"""

from __future__ import annotations

import os
import sys
import re
import subprocess
import datetime as dt
from typing import Dict, Any, Tuple, List;

import pandas as pd

# Lokaler Import – setzt voraus, dass das Arbeitsverzeichnis auf das
# Baseline-Repository gesetzt wurde (os.chdir in den Notebooks)
from script import inference


# ---------------------------------------------------------------------------
# Konstanten
# ---------------------------------------------------------------------------

# Output-Ordner der vier Challenge-Baselines (BL1–BL4).
# Ergebnisse dieser Ordner landen in der Baseline-CSV, nicht in der
# Experiment-CSV.
BASELINE_OUTPUT_DIRS = {
    "output_auto_avsr",
    "output_avsr_cocktail",
    "output_avsr_cocktail_finetuned",
    "output_muavic_en",
}

# Standard-Dateinamen für die Ergebnis-CSVs
BASELINE_CSV = "results_baseline_dev_without_central_videos.csv"
EXPERIMENT_CSV = "results_experiments_dev_without_central_videos.csv"

# ---------------------------------------------------------------------------
# Öffentliche Funktionen
# ---------------------------------------------------------------------------

def run_inference_for_experiment(
    exp_name: str,
    base_models: Dict[str, Dict[str, Any]],
    experiments: Dict[str, Dict[str, Any]],
    session_dir: str,
    extra_argv: list[str] | None = None,
) -> None:
    """
    Führt die Inference für genau EIN Experiment aus.

    Die Funktion baut sys.argv so zusammen, als würde inference.py direkt
    auf der Kommandozeile aufgerufen, und ruft dann inference.main() auf.
    So wird kein separater Subprozess gestartet und der GPU-Kontext bleibt
    im selben Prozess erhalten.

    Parameter
    ---------
    exp_name : str
        Schlüssel des Experiments in `experiments`, z.B. "E09_bs12_len20".
        Der Output-Ordner wird automatisch als "output_<exp_name>" gesetzt.

    base_models : dict
        Modell-Definitionen. Erwartet pro Eintrag:
            {
                "model_type": "<str>",           # z.B. "avsr_cocktail"
                "chkpt":      "<pfad oder id>",  # Checkpoint-Pfad oder HF-ID
            }

    experiments : dict
        Experiment-Konfigurationen. Erwartet pro Eintrag:
            {
                "base_model": "<key in base_models>",
                "beam_size":  <int>,
                "max_length": <int>,
                "comment":    "<optional>",
            }

    session_dir : str
        Glob-Pfad zu den Session-Verzeichnissen,
        z.B. "data-bin/dev/session_*".

    extra_argv : list[str] | None
        Optionale zusätzliche Kommandozeilenargumente die an inference.main()
        weitergegeben werden, z.B. ["--min_duration_on", "1.0"].
    """

    if exp_name not in experiments:
        raise ValueError(f"Experiment '{exp_name}' nicht in experiments dict gefunden.")

    cfg = experiments[exp_name]

    base_key = cfg.get("base_model")
    if base_key is None:
        raise ValueError(f"Experiment '{exp_name}' hat kein Feld 'base_model'.")

    if base_key not in base_models:
        raise ValueError(
            f"Experiment '{exp_name}' referenziert base_model '{base_key}', "
            f"aber das existiert nicht in base_models."
        )

    base = base_models[base_key]

    # Pflichtfelder auslesen
    try:
        beam_size = int(cfg["beam_size"])
        max_length = int(cfg["max_length"])
    except KeyError as e:
        raise ValueError(
            f"Experiment '{exp_name}' fehlt Pflichtfeld {e} ('beam_size' oder 'max_length')."
        ) from e

    model_type = base["model_type"]
    chkpt = base["chkpt"]
    out_dir_name = f"output_{exp_name}"

    print("\n=========================")
    print(f"Starte Inference für Experiment: {exp_name}")
    print(f"  base_model      = {base_key}")
    print(f"  model_type      = {model_type}")
    print(f"  checkpoint_path = {chkpt}")
    print(f"  beam_size       = {beam_size}")
    print(f"  max_length      = {max_length}")
    print(f"  output_dir_name = {out_dir_name}")
    print(f"  session_dir     = {session_dir}")

    comment = cfg.get("comment")
    if comment:
        print(f"  comment         = {comment}")

    # sys.argv simuliert einen Kommandozeilenaufruf von inference.py
    argv = [
        "notebook",  # Dummy-Programmname (wird von argparse ignoriert)
        "--model_type", model_type,
        "--session_dir", session_dir,
        "--checkpoint_path", chkpt,
        "--beam_size", str(beam_size),
        "--max_length", str(max_length),
        "--output_dir_name", out_dir_name,
    ]

    if extra_argv:
        argv.extend(extra_argv)

    sys.argv = argv

    # Einstiegspunkt aus script/inference.py aufrufen
    inference.main()




def _parse_evaluate_output(
    text: str,
    output_prefix: str = "output_",
) -> pd.DataFrame:
    """
    Parst den Summary-Block aus der Ausgabe von script/evaluate.py
    und gibt ein DataFrame mit Spalten:
        exp, avg_conv_f1, avg_speaker_wer, avg_joint_error, model
    zurück.
    """
    pattern = re.compile(
        r"Results for output dir prefix variant '([^']+)':\s+"
        r"\s*Average Conversation Clustering F1 score: ([0-9.]+)\s+"
        r"\s*Average Speaker WER: ([0-9.]+)\s+"
        r"\s*Average Joint ASR-Clustering Error Rate: ([0-9.]+)",
        re.MULTILINE,
    )

    rows = []
    for m in pattern.finditer(text):
        rows.append(
            {
                "exp": m.group(1),
                "avg_conv_f1": float(m.group(2)),
                "avg_speaker_wer": float(m.group(3)),
                "avg_joint_error": float(m.group(4)),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Präfix entfernen um reinen Modellnamen zu erhalten
    df["model"] = df["exp"].str.replace("^" + re.escape(output_prefix), "", regex=True)
    return df


def run_eval_and_log(
    session_dir: str,
    output_prefix: str = "output_",
    label_dir: str = "labels",
    baseline_csv: str | None = BASELINE_CSV,   # None = keine Baseline-CSV schreiben
    experiment_csv: str = EXPERIMENT_CSV,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Führt script/evaluate.py für eine Session aus, parst die Ausgabe
    und schreibt die Ergebnisse in die entsprechenden CSVs.

    Baseline-Modelle (BASELINE_OUTPUT_DIRS) werden in `baseline_csv`
    geschrieben – aber nur einmal, wenn die Datei noch nicht existiert.
    Alle anderen Modelle werden in `experiment_csv` gespeichert bzw.
    aktualisiert (letzter Wert gewinnt bei doppelten Experiment-Namen).

    Parameter
    ---------
    session_dir    : Pfad zum Session-Verzeichnis (oder Glob-Muster).
    output_prefix  : Gemeinsamer Präfix aller Output-Ordner ("output_").
    label_dir      : Name des Label-Unterordners innerhalb der Session.
    baseline_csv   : Pfad zur Baseline-CSV. None = nicht schreiben.
    experiment_csv : Pfad zur Experiment-CSV.

    Rückgabe
    --------
    full_df, baseline_df, experiments_df
    """


    cmd = [
    sys.executable, "script/evaluate.py",
    "--session_dir", session_dir,
    "--output_dir_name", output_prefix,
    "--label_dir_name", label_dir,
]


    print("Starte Evaluate:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Ausgabe in der Konsole anzeigen
    print(result.stdout)
    if result.stderr:
        print("=== STDERR evaluate.py ===")
        print(result.stderr)

    # Letzten Eval-Output zur Nachvollziehbarkeit sichern
    with open("eval_log_last_run.txt", "w", encoding="utf-8") as f:
        f.write(result.stdout)

    # Summary parsen
    full_df = _parse_evaluate_output(result.stdout, output_prefix=output_prefix)
    if full_df.empty:
        print("Keine Summary-Blöcke im Evaluate-Output gefunden.")
        return full_df, pd.DataFrame(), pd.DataFrame()

    # Baselines und Experimente trennen
    is_baseline = full_df["exp"].isin(BASELINE_OUTPUT_DIRS)
    baseline_df = full_df[is_baseline].copy()
    experiments_df = full_df[~is_baseline].copy()

    # Baseline-CSV: nur schreiben wenn noch nicht vorhanden
    if not baseline_df.empty and baseline_csv is not None:
        if not os.path.exists(baseline_csv):
            print(f"Schreibe Baselines nach '{baseline_csv}'")
            baseline_df.to_csv(
                baseline_csv,
                index=False,
                float_format="%.16g",
            )
        else:
            print(
                f"Baseline-CSV '{baseline_csv}' existiert bereits "
                f"(wird hier nicht überschrieben)."
            )

    # Experiment-CSV: anhängen/aktualisieren, letzter Wert gewinnt
    if not experiments_df.empty:
        experiments_df["timestamp"] = dt.datetime.now().isoformat(timespec="seconds")

        if os.path.exists(experiment_csv):
            old = pd.read_csv(experiment_csv)
            all_experiments = pd.concat([old, experiments_df], ignore_index=True)
            # Falls ein Experiment-Name mehrfach vorkommt: letzte Version behalten
            all_experiments = all_experiments.drop_duplicates(
                subset=["exp"], keep="last"
            )
        else:
            all_experiments = experiments_df

        print(f"Schreibe Experimente nach '{experiment_csv}'")
        all_experiments.to_csv(
            experiment_csv,
            index=False,
            float_format="%.16g",
        )
    else:
        print("Keine neuen Experimente in diesem Run (nur Baselines?).")

    return full_df, baseline_df, experiments_df


def append_eval_results_for_experiments(
    experiments: Dict[str, Dict[str, Any]],
    session_ids: List[str],
    target_csv: str = "results_dev_subset_by_session.csv",
    eval_script: str = "script/evaluate.py",
    session_dir_template: str = "data-bin/dev_without_central_videos/dev/{sid}",
    output_prefix: str = "output_",
    label_dir: str = "labels",
) -> pd.DataFrame:
    """
    Wertet mehrere Sessions aus und hängt die Ergebnisse session-basiert
    an eine gemeinsame CSV an.

    Jede Zeile in der CSV entspricht einer (session, model)-Kombination.
    Bereits vorhandene Kombinationen werden übersprungen (keine Duplikate).

    Parameter
    ---------
    experiments : dict
        Experiment-Configs. Die Keys müssen dem Modellnamen hinter "output_"
        entsprechen, z.B. "E09_bs12_len20" für Ordner "output_E09_bs12_len20".

    session_ids : list[str]
        Liste der Session-IDs, z.B. ["session_40", "session_43", ...].

    target_csv : str
        Pfad zur gemeinsamen Ergebnis-CSV (session-basiert).

    eval_script : str
        Pfad zu script/evaluate.py.

    session_dir_template : str
        Pfad-Template für Session-Verzeichnisse. {sid} wird durch die
        Session-ID ersetzt. Standard: "data-bin/dev/{sid}"
        (nach Umbenennung des Ordners, vormals
        "data-bin/dev_without_central_videos/dev/{sid}").

    output_prefix : str
        Gemeinsamer Präfix aller Output-Ordner, i.d.R. "output_".

    label_dir : str
        Name des Label-Unterordners innerhalb der Session.

    Rückgabe
    --------
    Vollständiges DataFrame mit allen Einträgen aus target_csv
    (inkl. neu angehängter Zeilen).
    """

    # Bestehende CSV laden oder leeres DataFrame anlegen
    if os.path.exists(target_csv):
        results_df = pd.read_csv(target_csv)
    else:
        results_df = pd.DataFrame(
            columns=[
                "session",
                "exp",
                "avg_conv_f1",
                "avg_speaker_wer",
                "avg_joint_error",
                "model",
                "timestamp",
            ]
        )

    # Bereits vorhandene (session, model)-Paare merken, um Duplikate zu vermeiden
    if not results_df.empty:
        existing_pairs = set(zip(results_df["session"], results_df["model"]))
    else:
        existing_pairs = set()

    new_rows = []

    for sid in session_ids:
        session_dir = session_dir_template.format(sid=sid)
        print(f"\n########## Evaluate für {sid} ##########")

        cmd = [
            sys.executable,
            eval_script,
            "--session_dir", session_dir,
            "--output_dir_name", output_prefix,
            "--label_dir_name", label_dir,
        ]

        print("Starte Evaluate:", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True)

        # Ausgabe optional zur Kontrolle
        print(result.stdout)
        if result.stderr:
            print("=== STDERR evaluate.py ===")
            print(result.stderr)

        # Summary für diesen Run parsen
        full_df = _parse_evaluate_output(result.stdout, output_prefix=output_prefix)
        if full_df.empty:
            print(f"Keine Summary-Blöcke im Evaluate-Output für {sid} gefunden.")
            continue

        # Nur Modelle aus `experiments` behalten
        full_df = full_df[full_df["model"].isin(experiments.keys())].copy()
        if full_df.empty:
            print(f"Keine der Varianten in {sid} passt zu den EXPERIMENTS-Keys.")
            continue

        ts = dt.datetime.now().isoformat(timespec="seconds")

        for _, row in full_df.iterrows():
            model_name = row["model"]
            exp_name = row["exp"]

            key = (sid, model_name)
            if key in existing_pairs:
                print(f"  -> Skippe bereits vorhandenes Ergebnis für {sid}, {model_name}")
                continue

            new_rows.append(
                {
                    "session": sid,
                    "exp": exp_name,
                    "avg_conv_f1": row["avg_conv_f1"],
                    "avg_speaker_wer": row["avg_speaker_wer"],
                    "avg_joint_error": row["avg_joint_error"],
                    "model": model_name,
                    "timestamp": ts,
                }
            )

    # Neue Zeilen anhängen und CSV schreiben
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        results_df = pd.concat([results_df, new_df], ignore_index=True)

        results_df.to_csv(
            target_csv,
            index=False,
            float_format="%.16g",
        )
        print(f"\n{len(new_rows)} neue Zeilen in '{target_csv}' geschrieben.")
    else:
        print("\nKeine neuen Zeilen – alles schon vorhanden oder keine passenden Modelle.")

    return results_df

# ---------------------------------------------------------------------------
# Öffentliche API
# ---------------------------------------------------------------------------

__all__ = [
    "BASELINE_OUTPUT_DIRS",
    "BASELINE_CSV",
    "EXPERIMENT_CSV",
    "run_inference_for_experiment",
    "run_eval_and_log",
    "append_eval_results_for_experiments",   
]

import os
import sys
import subprocess
from datetime import datetime

# Projektwurzel: .../team_c
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Wichtige Pfade
TEAMC_DATABIN = os.path.join(PROJECT_ROOT, "data-bin")
SCRIPT_DIR = os.path.join(PROJECT_ROOT, "script")
SEMANTIC_DIR = os.path.join(PROJECT_ROOT, "src", "semantic")

# MCoRec dev-Daten (wie in main.py verwendet)
MCOREC_DEV_DIR = os.path.join(PROJECT_ROOT, "../../../mcorec_baseline/data-bin/dev")


def run(cmd, cwd=None, capture=False):
    """Hilfsfunktion für subprocess-Aufrufe."""
    print(f"\n[RUN] ({cwd or os.getcwd()}) $ {' '.join(cmd)}")
    if capture:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(result.stdout)
        return result.stdout
    else:
        subprocess.run(cmd, cwd=cwd, check=True)


def main():
    """
    End-to-End-Pipeline:

    1) Texte aus den .vtt-Dateien extrahieren → dev_texts.json
    2) Sentence-Embeddings erzeugen → E.npy, ids.json
    3) Cosine-Distanzmatrix bauen → D.npy
    4) Zeit / Semantik / Hybrid-Clustering für alle Sessions → speaker_to_cluster_*.json
    5) Evaluation aller Sessions → F1 / ARI
    6) Evaluationsergebnis mit Zeitstempel in Datei schreiben
    """

    os.makedirs(TEAMC_DATABIN, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1) Texte extrahieren (überschreibt dev_texts.json)
    #    Script: src/semantic/extract_texts.py
    # ------------------------------------------------------------------ #
    dev_texts_path = os.path.join(TEAMC_DATABIN, "dev_texts.json")

    run(
        [
            "python",
            "extract_texts.py",
            "--input_dir",
            MCOREC_DEV_DIR,
            "--output_json",
            dev_texts_path,
        ],
        cwd=SEMANTIC_DIR,
    )

    # ------------------------------------------------------------------ #
    # 2) Embeddings erzeugen (überschreibt E.npy / ids.json)
    #    Script: src/semantic/embed_texts.py
    # ------------------------------------------------------------------ #
    run(
        [
            "python",
            "embed_texts.py",
            "--input_json",
            dev_texts_path,
            "--outdir",
            TEAMC_DATABIN,
            "--normalize",
        ],
        cwd=SEMANTIC_DIR,
    )

    # ------------------------------------------------------------------ #
    # 3) Distanzmatrix bauen (überschreibt D.npy)
    #    Script: src/semantic/build_distance_matrix.py
    # ------------------------------------------------------------------ #
    run(
        [
            "python",
            "build_distance_matrix.py",
            "--dir",
            TEAMC_DATABIN,
            "--overwrite",
        ],
        cwd=SEMANTIC_DIR,
    )

    # ------------------------------------------------------------------ #
    # 4) Clustering: Zeit / Semantik / Hybrid für alle Sessions
    #    Script: script/main.py
    # ------------------------------------------------------------------ #
    run(
        [
            "python",
            "main.py",
            "--session_dir",
            "../../../mcorec_baseline/data-bin/dev/session_*",
            "--output_dir_name",
            "output_semantic_test",
        ],
        cwd=SCRIPT_DIR,
    )

    # ------------------------------------------------------------------ #
    # 5) Evaluation aller Sessions (nutzt eval_sessions.py)
    #    → Ausgabe einsammeln
    # ------------------------------------------------------------------ #
    eval_output = run(
        [
            "python",
            "eval_sessions.py",
        ],
        cwd=SCRIPT_DIR,
        capture=True,
    )

    # ------------------------------------------------------------------ #
    # 6) Evaluation mit Zeitstempel persistieren
    # ------------------------------------------------------------------ #
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(TEAMC_DATABIN, "reports")
    os.makedirs(report_dir, exist_ok=True)

    report_path = os.path.join(report_dir, f"evaluation_{ts}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(eval_output)

    print(f"\n[OK] Evaluation report geschrieben nach:\n  {report_path}")


if __name__ == "__main__":
    main()

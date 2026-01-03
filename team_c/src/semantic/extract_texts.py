"""
Extrahiert Text aus den *System*-Hypothesen (.vtt) der CHiME-9 MCoRec Baseline
und schreibt sie in ein einheitliches JSON-Format mit IDs + Texten.

Wichtig:
- Liest NICHT aus `labels/` (Ground Truth), sondern aus `output/` (Baseline-Hypothesen).
- baseline_dir kann entweder
  (a) ein Split-Ordner sein (enthält session_*), oder
  (b) ein einzelner Session-Ordner (z. B. .../session_40).

Beispiel:
python extract_texts.py \
  --baseline_dir /home/ercel001/AUVIS/task5_semantic_prototype/auvis/team_c/data-bin/baseline/dev \
  --output_json  /home/ercel001/AUVIS/task5_semantic_prototype/auvis/team_c/data-bin/_output/dev/semantik_clustering/dev_texts.json
"""

import os
import json
import argparse
from typing import List, Dict


def read_vtt_file(path: str) -> List[str]:
    """
    Liest eine .vtt-Datei ein und gibt Textzeilen zurück (ohne Zeitstempel und Header).
    """
    lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # Header / Metadaten
            if line.startswith("WEBVTT"):
                continue
            if line.startswith(("NOTE", "STYLE", "REGION")):
                continue
            if "X-TIMESTAMP-MAP" in line:
                continue

            # Zeitstempelzeile
            if "-->" in line:
                continue

            # Cue-ID (kommt je nach Generator vor)
            if line.isdigit():
                continue

            lines.append(line)
    return lines


def _list_sessions(baseline_dir: str) -> List[str]:
    """
    Liefert eine Liste von Session-Pfaden.
    Unterstützt:
    - Split-Ordner: .../dev (enthält session_*)
    - Session-Ordner: .../dev/session_40
    """
    if not os.path.isdir(baseline_dir):
        raise FileNotFoundError(f"baseline_dir existiert nicht oder ist kein Ordner: {baseline_dir}")

    base_name = os.path.basename(os.path.normpath(baseline_dir))
    if base_name.startswith("session_"):
        return [baseline_dir]

    sessions = []
    for s in sorted(os.listdir(baseline_dir)):
        p = os.path.join(baseline_dir, s)
        if s.startswith("session_") and os.path.isdir(p):
            sessions.append(p)
    return sessions


def collect_texts(baseline_dir: str, output_dir_name: str = "output") -> Dict[str, List[str]]:
    """
    Extrahiert pro Session und pro Sprecher die Hypothesen-Transkripte aus:
      <session>/<output_dir_name>/*.vtt
    """
    all_ids: List[str] = []
    all_texts: List[str] = []

    session_paths = _list_sessions(baseline_dir)

    for session_path in session_paths:
        session_name = os.path.basename(os.path.normpath(session_path))
        output_path = os.path.join(session_path, output_dir_name)
        if not os.path.isdir(output_path):
            continue

        vtts = sorted(
            f for f in os.listdir(output_path)
            if f.startswith("spk_") and f.endswith(".vtt")
        )

        for fn in vtts:
            speaker_id = fn[:-4]  # "spk_0"
            file_path = os.path.join(output_path, fn)

            texts = read_vtt_file(file_path)
            if not texts:
                continue

            joined = " ".join(texts).strip()
            if not joined:
                continue

            uid = f"{session_name}_{speaker_id}"
            all_ids.append(uid)
            all_texts.append(joined)

    print(f"✓ {len(all_texts)} Sprecher-Texte extrahiert (aus '{output_dir_name}/').")
    return {"ids": all_ids, "texts": all_texts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--baseline_dir",
        required=True,
        help="Split-Ordner (.../dev) ODER einzelner Session-Ordner (.../dev/session_40).",
    )
    parser.add_argument(
        "--output_json",
        required=True,
        help="Zielpfad für die JSON-Datei.",
    )
    parser.add_argument(
        "--output_dir_name",
        default="output",
        help="Name des System-Output-Ordners pro Session (default: output).",
    )
    args = parser.parse_args()

    result = collect_texts(args.baseline_dir, output_dir_name=args.output_dir_name)

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"→ JSON gespeichert unter: {args.output_json}")


if __name__ == "__main__":
    main()

"""
Extrahiert Text aus den .vtt-Dateien der CHiME-9 MCoRec Challenge (z. B. dev-Split)
und schreibt sie in ein einheitliches JSON-Format mit IDs + Texten.

Beispiel:
python extract_texts.py \
  --input_dir /home/ercel001/AUVIS/mcorec_baseline/data-bin/dev \
  --output_json /home/ercel001/AUVIS/task5_semantic_prototype/auvis/team_c/data-bin/dev_texts.json
"""

import os
import json
import argparse

def read_vtt_file(path: str) -> list[str]:
    """Liest eine .vtt-Datei ein und gibt alle Textzeilen (ohne Zeitstempel) zurück."""
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("WEBVTT") or "-->" in line:
                continue
            lines.append(line)
    return lines

def collect_texts(input_dir: str) -> dict:
    """
    Durchläuft alle Sessions unterhalb von input_dir und extrahiert pro Sprecher die Transkripte.
    Gibt ein Dict mit Listen 'ids' und 'texts' zurück.
    """
    all_ids, all_texts = [], []
    for session in sorted(os.listdir(input_dir)):
        session_path = os.path.join(input_dir, session)
        labels_path = os.path.join(session_path, "labels")
        if not os.path.isdir(labels_path):
            continue

        for file in sorted(os.listdir(labels_path)):
            if not file.endswith(".vtt"):
                continue
            speaker_id = file.replace(".vtt", "")
            file_path = os.path.join(labels_path, file)

            texts = read_vtt_file(file_path)
            if not texts:
                continue

            joined_text = " ".join(texts)
            uid = f"{session}_{speaker_id}"
            all_ids.append(uid)
            all_texts.append(joined_text)

    print(f"✓ {len(all_texts)} Sprecher-Texte extrahiert.")
    return {"ids": all_ids, "texts": all_texts}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Pfad zum dev- oder train-Verzeichnis (z. B. data-bin/dev)")
    parser.add_argument("--output_json", required=True, help="Zielpfad für die JSON-Datei")
    args = parser.parse_args()

    result = collect_texts(args.input_dir)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"→ JSON gespeichert unter: {args.output_json}")

if __name__ == "__main__":
    main()

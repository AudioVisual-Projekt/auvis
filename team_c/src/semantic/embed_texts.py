# embed_texts.py
import argparse
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import trange
import torch


def load_items(path: str):
    """Unterstützt drei Schemata:
       1) Liste von Dicts: [{id:.., text:..}, ...]
       2) Dict mit 'items': {"items": [...]}
       3) Dict mit 'ids' und 'texts': {"ids":[...], "texts":[...]}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "items" in data and isinstance(data["items"], list):
        return data["items"]
    if isinstance(data, dict) and "ids" in data and "texts" in data:
        ids, texts = data["ids"], data["texts"]
        assert isinstance(ids, list) and isinstance(texts, list) and len(ids) == len(texts), \
            "ids/texts müssen Listen gleicher Länge sein"
        return [{"id": str(i), "text": str(t)} for i, t in zip(ids, texts)]

    raise ValueError("Unerwartetes JSON-Format (Liste | Dict['items'] | Dict['ids','texts']).")


def pick_id(d):
    return str(d.get("id"))


def pick_text(d):
    return str(d.get("text", "")).strip()


def main():
    ap = argparse.ArgumentParser(description="Erzeuge Sentence-Embeddings aus Transkripten (CHiME-9)")
    ap.add_argument("--input_json", required=True, help="Pfad zur JSON mit ids/texts")
    ap.add_argument("--outdir", required=True, help="Zielverzeichnis (z. B. .../_output/dev/semantik_clustering)")
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2",
                    help="Sentence-Transformer Modellname oder lokaler Pfad")
    ap.add_argument("--batch", type=int, default=1024, help="Batchgröße (GPU-abhängig)")
    ap.add_argument("--normalize", action="store_true", help="Embeddings auf L2-Norm=1 normalisieren")
    ap.add_argument("--prefix", default="",
                    help="Optionaler Dateiprefix (z. B. 'dev_') für zusätzliche Dateien")
    args = ap.parse_args()

    assert os.path.isfile(args.input_json), f"Datei nicht gefunden: {args.input_json}"
    os.makedirs(args.outdir, exist_ok=True)

    print(f"→ Lade Items aus: {args.input_json}")
    items = load_items(args.input_json)

    pairs = [(pick_id(d), pick_text(d)) for d in items if pick_text(d)]
    if not pairs:
        raise ValueError("Keine gültigen Text-Einträge gefunden.")

    ids, texts = zip(*pairs)
    print(f"→ Einträge (nach Filter): {len(texts)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"→ Lade Modell: {args.model} auf {device}")
    model = SentenceTransformer(args.model, device=device)

    embs = []
    for s in trange(0, len(texts), args.batch, desc="Embeddings"):
        batch = texts[s:s + args.batch]
        embs.append(model.encode(
            batch,
            batch_size=len(batch),
            convert_to_numpy=True,
            normalize_embeddings=args.normalize,
            show_progress_bar=False
        ))
    E = np.vstack(embs)

    # Standard-Artefakte (werden downstream erwartet)
    e_std = os.path.join(args.outdir, "E.npy")
    ids_std = os.path.join(args.outdir, "ids.json")

    np.save(e_std, E)
    with open(ids_std, "w", encoding="utf-8") as f:
        json.dump(list(ids), f, ensure_ascii=False, indent=2)

    # Optional: zusätzliche prefixed Kopien (z. B. für Vergleichsläufe)
    if args.prefix:
        e_pref = os.path.join(args.outdir, f"{args.prefix}E.npy")
        ids_pref = os.path.join(args.outdir, f"{args.prefix}ids.json")
        np.save(e_pref, E)
        with open(ids_pref, "w", encoding="utf-8") as f:
            json.dump(list(ids), f, ensure_ascii=False, indent=2)

    print(f"✓ Gespeichert: {e_std}  Shape={E.shape}")
    print(f"✓ Gespeichert: {ids_std}  N={len(ids)}")
    if args.prefix:
        print(f"✓ Zusätzlich: {e_pref}")
        print(f"✓ Zusätzlich: {ids_pref}")


if __name__ == "__main__":
    main()

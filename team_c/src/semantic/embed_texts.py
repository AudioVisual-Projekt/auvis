# embed_texts.py (final English-only version)
import argparse, json, os
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

def pick_id(d):   return str(d.get("id"))
def pick_text(d): return str(d.get("text", "")).strip()

def main():
    ap = argparse.ArgumentParser(description="Erzeuge englische Sentence-Embeddings aus Transkripten (CHiME-9)")
    ap.add_argument("--input_json", required=True, help="Pfad zur dev_texts.json")
    ap.add_argument("--outdir", required=True, help="Zielverzeichnis (z. B. ../../data-bin)")
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2",
                    help="Englisches High-Quality-Modell (768-D)")
    ap.add_argument("--batch", type=int, default=1024, help="Batchgröße (GPU-abhängig)")
    ap.add_argument("--normalize", action="store_true", help="Embeddings auf L2-Norm=1 normalisieren")
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
        batch = texts[s:s+args.batch]
        embs.append(model.encode(
            batch,
            batch_size=len(batch),
            convert_to_numpy=True,
            normalize_embeddings=args.normalize,
            show_progress_bar=False
        ))
    E = np.vstack(embs)

    e_path  = os.path.join(args.outdir, "E.npy")
    ids_path = os.path.join(args.outdir, "ids.json")
    np.save(e_path, E)
    with open(ids_path, "w", encoding="utf-8") as f:
        json.dump(list(ids), f, ensure_ascii=False, indent=2)

    print(f"✓ Gespeichert: {e_path}  Shape={E.shape}")
    print(f"✓ Gespeichert: {ids_path}  N={len(ids)}")

if __name__ == "__main__":
    main()

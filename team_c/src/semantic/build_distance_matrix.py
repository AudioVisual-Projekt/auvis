"""
build_distance_matrix.py

Zweck
-----
Aus einer Embedding-Matrix E.npy (N x d) wird eine Distanzmatrix D.npy (N x N) gebaut.

- N = Anzahl Sprechertexte (z. B. session_40_spk_0, session_40_spk_1, ...)
- d = Embedding-Dimension (z. B. 768)

Distanzmaß
----------
Wir verwenden Cosine-Distanz:
    dist(i,j) = 1 - cosine_similarity(E[i], E[j])

Warum blockweise?
-----------------
Eine NxN-Matrix wird schnell groß. Wir berechnen daher zeilenweise in Blöcken, um
Speicherverbrauch und Peak-RAM zu reduzieren.
"""

import argparse
import os
import numpy as np
from numpy.linalg import norm

try:
    from tqdm import trange
except ImportError:
    def trange(*args, **kwargs):
        return range(*args)


def compute_blockwise_cosine_distance(
    E: np.ndarray,
    block_size: int = 512,
    eps: float = 1e-12
) -> np.ndarray:
    """
    Berechnet die Cosine-Distanzmatrix D (N x N) aus E (N x d) blockweise.

    Cosine Similarity:
        cos_sim(a,b) = (a·b) / (||a|| * ||b||)

    Cosine Distanz:
        cos_dist(a,b) = 1 - cos_sim(a,b)
    """
    # Wir arbeiten in float32, um Speicher zu sparen und trotzdem stabil zu bleiben.
    E = E.astype(np.float32, copy=False)
    N = E.shape[0]

    # Norm jedes Embedding-Vektors (Shape: N x 1).
    norms = norm(E, axis=1, keepdims=True).astype(np.float32)

    # Ergebnis-Matrix (N x N).
    D = np.empty((N, N), dtype=np.float32)

    # Blockweise: wir berechnen jeweils einen Zeilenblock (B x N).
    for start in trange(0, N, block_size, desc="Building distance matrix"):
        end = min(start + block_size, N)

        # E_block: (B x d), norms_block: (B x 1)
        E_block = E[start:end]
        norms_block = norms[start:end]

        # Skalarprodukte: (B x N)
        sim = E_block @ E.T

        # Nenner: (B x N) = (B x 1) @ (1 x N)
        denom = norms_block @ norms.T

        # Cosine Similarity
        sim = sim / (denom + eps)

        # Cosine Distanz
        D[start:end, :] = (1.0 - sim).astype(np.float32)

    # Selbst-Distanz ist per Definition 0.
    np.fill_diagonal(D, 0.0)

    # Numerische Symmetrie-Stabilisierung.
    D = 0.5 * (D + D.T)

    return D.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Baue eine Cosine-Distanzmatrix (D.npy) aus E.npy (blockweise)."
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        required=True,
        help="Arbeitsverzeichnis, z. B. .../data-bin/_output/dev/semantik_clustering"
    )
    parser.add_argument(
        "--emb_file",
        type=str,
        default="E.npy",
        help="Dateiname der Embedding-Matrix (Standard: E.npy)."
    )
    parser.add_argument(
        "--out_file",
        type=str,
        default="D.npy",
        help="Zieldatei für die Distanzmatrix (Standard: D.npy)."
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=512,
        help="Blockgröße (Anzahl Zeilen pro Block)."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Existierende Ausgabedatei überschreiben, falls vorhanden."
    )
    args = parser.parse_args()

    # Pfad normalisieren: ~ expanden, dann absolut machen.
    work_dir = os.path.abspath(os.path.expanduser(args.work_dir))
    emb_path = os.path.join(work_dir, args.emb_file)
    out_path = os.path.join(work_dir, args.out_file)

    if not os.path.isfile(emb_path):
        raise FileNotFoundError(f"Embedding-Datei nicht gefunden: {emb_path}")

    if os.path.exists(out_path) and not args.overwrite:
        raise FileExistsError(
            f"{out_path} existiert bereits. Starte mit --overwrite, um zu überschreiben."
        )

    print(f"→ Lade Embeddings aus: {emb_path}")
    E = np.load(emb_path)

    # Schutz: E muss 2D sein (N x d).
    if E.ndim != 2:
        raise ValueError(f"E muss 2D sein (N x d). Gefunden: shape={E.shape}")

    print(f"   Shape von E: {E.shape}")
    print(f"→ Berechne Cosine-Distanzen (block_size={args.block_size}) …")
    D = compute_blockwise_cosine_distance(E, block_size=args.block_size)

    print(f"→ Speichere Distanzmatrix nach: {out_path}")
    np.save(out_path, D)

    print(f"✓ Fertig. D-Shape: {D.shape}")


if __name__ == "__main__":
    main()

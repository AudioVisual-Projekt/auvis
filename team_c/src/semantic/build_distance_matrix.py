import argparse
import os
import numpy as np
from numpy.linalg import norm

try:
    from tqdm import trange
except ImportError:
    # Fallback, falls tqdm nicht installiert ist
    def trange(*args, **kwargs):
        return range(*args)


def compute_blockwise_cosine_distance(E: np.ndarray,
                                      block_size: int = 512,
                                      eps: float = 1e-12) -> np.ndarray:
    """
    Berechnet eine vollständige Cosine-Distanzmatrix D aus einer
    Embedding-Matrix E (Shape: N x D) blockweise über die Zeilen.

    D[i, j] = 1 - cos_sim(E[i], E[j])

    Args:
        E: Embedding-Matrix (N x D)
        block_size: Anzahl der Zeilen, die pro Block verarbeitet werden
        eps: Numerischer Schutzterm im Nenner

    Returns:
        D: Cosine-Distanzmatrix (N x N), dtype=float32
    """
    # Auf float32 casten, um Speicher zu sparen (optional)
    E = E.astype(np.float32, copy=False)

    N = E.shape[0]
    # Normen aller Vektoren einmalig berechnen
    norms = norm(E, axis=1, keepdims=True).astype(np.float32)

    # Zielmatrix
    D = np.empty((N, N), dtype=np.float32)

    for start in trange(0, N, block_size, desc="Building distance matrix"):
        end = min(start + block_size, N)

        E_block = E[start:end]                  # Shape: (B x D)
        norms_block = norms[start:end]          # Shape: (B x 1)

        # Cosine Similarity: (E_block @ E.T) / (norms_block * norms.T)
        sim = E_block @ E.T                     # (B x N)
        denom = norms_block @ norms.T           # (B x N)
        sim = sim / (denom + eps)

        # Cosine-Distanz
        dist_block = 1.0 - sim

        # In Zieldistanzmatrix schreiben
        D[start:end, :] = dist_block.astype(np.float32)

    # Diagonale exakt auf 0 setzen
    np.fill_diagonal(D, 0.0)

    # Optionale Symmetrisierung (nur zur Sicherheit gegen Rundungsfehler)
    D = 0.5 * (D + D.T)

    return D.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Baue eine Cosine-Distanzmatrix D.npy aus E.npy (blockweise)."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="../../data-bin",
        help="Verzeichnis, in dem E.npy liegt und D.npy gespeichert werden soll."
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
        help="Existierende D.npy überschreiben, falls vorhanden."
    )

    args = parser.parse_args()

    emb_path = os.path.join(args.dir, args.emb_file)
    out_path = os.path.join(args.dir, args.out_file)

    assert os.path.isfile(emb_path), f"E.npy nicht gefunden unter: {emb_path}"

    if os.path.exists(out_path) and not args.overwrite:
        raise FileExistsError(
            f"{out_path} existiert bereits. "
            f"Starte mit --overwrite, um zu überschreiben."
        )

    print(f"→ Lade Embeddings aus: {emb_path}")
    E = np.load(emb_path)
    print(f"   Shape von E: {E.shape}")

    print(f"→ Berechne Cosine-Distanzen (block_size={args.block_size}) …")
    D = compute_blockwise_cosine_distance(E, block_size=args.block_size)

    print(f"→ Speichere Distanzmatrix nach: {out_path}")
    np.save(out_path, D)
    print("✓ Fertig. D-Shape:", D.shape)


if __name__ == "__main__":
    main()

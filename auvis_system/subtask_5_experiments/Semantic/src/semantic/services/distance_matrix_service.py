"""
distance_matrix_service.py

Ziel:
Testbare Kernlogik für Pipeline-Schritt 3 (Distanzmatrix je Session):
Aus Embeddings (np.ndarray oder embeddings.npy) + IDs (embedding_ids.json) wird eine Cosine-Distanzmatrix
(DistanceMatrixSessionArtifacts) berechnet.

Wichtige Eigenschaften
----------------------
- Keine CLI / kein argparse
- Keine hart kodierten Pfade
- Reine Service-API (E + ids oder Dateipfade)
- Robust: Validierung, Symmetrie/Diagonal-Checks, optionaler Fast-Path bei L2-normalisierten Embeddings
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from numpy.linalg import norm

from semantic.models.distance_matrix_models import DistanceMatrixSessionArtifacts, DistanceMeta


def _get_trange():
    """
    Liefert trange() von tqdm, falls verfügbar; andernfalls range()-Fallback.
    """
    try:
        from tqdm import trange  # type: ignore
        return trange
    except Exception:
        def trange(*args, **kwargs):  # noqa: ANN001
            return range(*args)
        return trange


def _load_json_list(path: Union[str, Path]) -> list[dict[str, Any]]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"JSON-Datei nicht gefunden: {p}")
    obj = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError(f"Erwartet JSON-Liste in {p}, gefunden: {type(obj)}")
    # Typ leicht härten (die Models validieren Details)
    return obj  # type: ignore[return-value]


def _is_l2_normalized(
    E: np.ndarray,
    *,
    atol: float = 1e-3,
) -> bool:
    """
    Prüft, ob Zeilen von E näherungsweise L2-normalisiert sind (||e_i|| ~= 1).
    """
    if E.size == 0:
        return True
    nrm = norm(E, axis=1)
    if not np.isfinite(nrm).all():
        return False
    return bool(np.all(np.abs(nrm - 1.0) <= atol))


def compute_blockwise_cosine_distance(
    E: np.ndarray,
    *,
    block_size: int = 512,
    eps: float = 1e-12,
    show_progress: bool = False,
    assume_normalized: Optional[bool] = None,
    normalized_atol: float = 1e-3,
) -> tuple[np.ndarray, bool, dict[str, Any]]:
    """
    Berechnet die Cosine-Distanzmatrix D (N x N) aus E (N x d) blockweise.

    Rückgabe
    --------
    (D, embeddings_normed, numeric_info)
    """
    if not isinstance(E, np.ndarray):
        raise TypeError("E muss ein numpy.ndarray sein.")
    if E.ndim != 2:
        raise ValueError(f"E muss 2D sein (N x d). Gefunden: shape={E.shape}")
    if block_size <= 0:
        raise ValueError("block_size muss > 0 sein.")
    if eps <= 0:
        raise ValueError("eps muss > 0 sein.")

    # float32 reicht in der Regel aus und spart RAM
    E = E.astype(np.float32, copy=False)
    N = E.shape[0]

    if assume_normalized is None:
        embeddings_normed = _is_l2_normalized(E, atol=normalized_atol)
    else:
        embeddings_normed = bool(assume_normalized)

    D = np.empty((N, N), dtype=np.float32)

    # Nur berechnen, wenn notwendig (spart Zeit bei normierten Embeddings)
    norms: Optional[np.ndarray] = None
    if not embeddings_normed:
        norms = norm(E, axis=1, keepdims=True).astype(np.float32)

    trange = _get_trange()
    iterator = trange(0, N, block_size, desc="Building distance matrix") if show_progress else range(0, N, block_size)

    for start in iterator:
        end = min(start + block_size, N)
        E_block = E[start:end]  # (B x d)

        # Skalarprodukte: (B x N)
        sim = E_block @ E.T

        if not embeddings_normed:
            assert norms is not None
            norms_block = norms[start:end]          # (B x 1)
            denom = norms_block @ norms.T           # (B x N)
            sim = sim / (denom + eps)
        # else: normiert -> cosine sim = dot product

        D[start:end, :] = (1.0 - sim).astype(np.float32)

    # Diagonale exakt 0
    np.fill_diagonal(D, 0.0)

    # Numerische Symmetrie-Stabilisierung
    D = 0.5 * (D + D.T)

    # Kleine numerische Unterläufe leicht korrigieren (ohne grobes Clipping)
    # (Cosine-Distanz ist i. d. R. >= 0)
    D[D < 0.0] = 0.0

    # Numerik-Infos für Meta/Debug
    diag = np.diag(D) if N > 0 else np.array([], dtype=np.float32)
    numeric_info: dict[str, Any] = {
        "dtype": str(D.dtype),
        "block_size": int(block_size),
        "eps": float(eps),
        "embeddings_normed": bool(embeddings_normed),
        "diag_max_abs": float(np.max(np.abs(diag))) if diag.size else 0.0,
        "sym_max_abs_diff": float(np.max(np.abs(D - D.T))) if N > 0 else 0.0,
        "min": float(np.min(D)) if N > 0 else 0.0,
        "max": float(np.max(D)) if N > 0 else 0.0,
        "mean": float(np.mean(D)) if N > 0 else 0.0,
    }

    return D.astype(np.float32, copy=False), embeddings_normed, numeric_info


def build_distance_matrix_session(
    *,
    E: np.ndarray,
    ids: list[dict[str, Any]],
    distance_type: str = "cosine",
    block_size: int = 512,
    eps: float = 1e-12,
    show_progress: bool = False,
    assume_normalized: Optional[bool] = None,
    normalized_atol: float = 1e-3,
    input_fingerprint: Optional[str] = None,
    model_name: Optional[str] = None,
    strict_numeric_checks: bool = True,
) -> DistanceMatrixSessionArtifacts:
    """
    Erstellt Distanz-Artefakte für eine Session aus (E, ids).
    """
    if distance_type != "cosine":
        raise ValueError(f"Unsupported distance_type: {distance_type!r}. Aktuell unterstützt: 'cosine'.")

    D, embeddings_normed, numeric_info = compute_blockwise_cosine_distance(
        E,
        block_size=block_size,
        eps=eps,
        show_progress=show_progress,
        assume_normalized=assume_normalized,
        normalized_atol=normalized_atol,
    )

    meta = DistanceMeta(
        distance_type=distance_type,
        n=int(D.shape[0]),
        input_fingerprint=input_fingerprint,
        model_name=model_name,
        embeddings_normed=bool(embeddings_normed),
        numeric=numeric_info,
    )

    artifacts = DistanceMatrixSessionArtifacts(distances=D, ids=ids, meta=meta)
    artifacts.validate(strict_numeric_checks=strict_numeric_checks)
    return artifacts


def build_distance_matrix_session_from_files(
    *,
    embeddings_path: Union[str, Path],
    ids_path: Union[str, Path],
    distance_type: str = "cosine",
    block_size: int = 512,
    eps: float = 1e-12,
    mmap_mode: Optional[str] = None,
    show_progress: bool = False,
    assume_normalized: Optional[bool] = None,
    normalized_atol: float = 1e-3,
    input_fingerprint: Optional[str] = None,
    model_name: Optional[str] = None,
    strict_numeric_checks: bool = True,
) -> DistanceMatrixSessionArtifacts:
    """
    Convenience-API: lädt E + ids aus Dateien und erzeugt Session-Artefakte.
    """
    emb_p = Path(embeddings_path).expanduser().resolve()
    if not emb_p.is_file():
        raise FileNotFoundError(f"Embedding-Datei nicht gefunden: {emb_p}")
    E = np.load(emb_p, mmap_mode=mmap_mode)

    ids = _load_json_list(ids_path)

    return build_distance_matrix_session(
        E=E,
        ids=ids,
        distance_type=distance_type,
        block_size=block_size,
        eps=eps,
        show_progress=show_progress,
        assume_normalized=assume_normalized,
        normalized_atol=normalized_atol,
        input_fingerprint=input_fingerprint,
        model_name=model_name,
        strict_numeric_checks=strict_numeric_checks,
    )

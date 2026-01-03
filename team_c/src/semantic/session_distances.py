# src/semantic/session_distances.py
import os
import json
from typing import List, Dict, Tuple
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=8)
def _load_ids_and_dist_matrix(
    data_dir: str,
    ids_file: str = "ids.json",
    dist_file: str = "D.npy",
) -> Tuple[Tuple[str, ...], np.ndarray]:
    """
    Lädt die globale ID-Liste und die globale Distanzmatrix.

    Wichtige Optimierungen:
    - Caching: verhindert, dass ids.json und D.npy pro Session neu geladen werden.
    - mmap_mode="r": D.npy wird gemappt (nicht komplett in RAM geladen), was bei großen Matrizen
      typischerweise zwingend ist.

    Erwartung:
    - ids.json: List[str], z. B. ["session_40_spk_0", ...]
    - D.npy:    np.ndarray mit Shape (N, N)
    """
    data_dir = os.path.abspath(os.path.expanduser(data_dir))

    ids_path = os.path.join(data_dir, ids_file)
    dist_path = os.path.join(data_dir, dist_file)

    if not os.path.isfile(ids_path):
        raise FileNotFoundError(f"ids.json nicht gefunden: {ids_path}")
    if not os.path.isfile(dist_path):
        raise FileNotFoundError(f"Distanzmatrix nicht gefunden: {dist_path}")

    with open(ids_path, "r", encoding="utf-8") as f:
        ids = json.load(f)

    if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
        raise ValueError(f"{ids_path} hat nicht das erwartete Format List[str].")

    # mmap_mode="r" => große Datei nicht vollständig in RAM laden
    D = np.load(dist_path, mmap_mode="r")

    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"{dist_path} muss eine quadratische Matrix sein. D.shape={D.shape}")

    if len(ids) != D.shape[0]:
        raise ValueError(
            f"ids/D nicht konsistent: len(ids)={len(ids)}, D.shape={D.shape}"
        )

    # ids als Tuple zurückgeben, damit es hashbar ist (für lru_cache)
    return tuple(ids), D


def _build_index_map(ids: Tuple[str, ...]) -> Dict[str, int]:
    """
    Mappt jede ID (z. B. 'session_40_spk_0') auf ihren Index in D.
    """
    return {id_str: i for i, id_str in enumerate(ids)}


def get_session_semantic_distance_matrix(
    data_dir: str,
    session_name: str,
    speaker_names: List[str],
    ids_file: str = "ids.json",
    dist_file: str = "D.npy",
) -> Tuple[np.ndarray, List[str]]:
    """
    Gibt die semantische Distanzmatrix für EINE Session zurück.

    Beispiel (dein Schema):
    - session_name = "session_40"
    - speaker_names = ["spk_0", "spk_1", ...]

    Daraus werden IDs gebaut wie:
    - "session_40_spk_0", "session_40_spk_1", ...
    """
    ids, D = _load_ids_and_dist_matrix(data_dir, ids_file, dist_file)
    index_map = _build_index_map(ids)

    indices: List[int] = []
    full_ids: List[str] = []

    for spk in speaker_names:
        full_id = f"{session_name}_{spk}"
        idx = index_map.get(full_id)
        if idx is None:
            raise KeyError(
                f"ID '{full_id}' nicht in {ids_file} gefunden. "
                f"Beispiele: {list(ids[:5])} ..."
            )
        indices.append(idx)
        full_ids.append(full_id)

    idx_arr = np.asarray(indices, dtype=int)
    D_session = np.asarray(D[np.ix_(idx_arr, idx_arr)], dtype=np.float32)

    return D_session, full_ids

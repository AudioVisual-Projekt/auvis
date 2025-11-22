import os
import json
from typing import List, Dict, Tuple

import numpy as np


def _load_ids_and_dist_matrix(
    data_dir: str,
    ids_file: str = "ids.json",
    dist_file: str = "D.npy",
) -> Tuple[List[str], np.ndarray]:
    """
    Lädt die globale ID-Liste und die globale Distanzmatrix.

    Erwartung:
    - ids.json: List[str], z. B. ["dev0001_SPK01", "dev0001_SPK02", ...]
    - D.npy:    np.ndarray mit Shape (N, N)
    """
    ids_path = os.path.join(data_dir, ids_file)
    dist_path = os.path.join(data_dir, dist_file)

    with open(ids_path, "r", encoding="utf-8") as f:
        ids = json.load(f)

    D = np.load(dist_path)

    assert len(ids) == D.shape[0] == D.shape[1], (
        f"ids/D nicht konsistent: len(ids)={len(ids)}, "
        f"D.shape={D.shape}"
    )

    return ids, D


def _build_index_map(ids: List[str]) -> Dict[str, int]:
    """
    Mappt jede ID (z. B. 'dev0001_SPK01') auf ihren Index in D.
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

    data_dir:
        Pfad, in dem ids.json und D.npy liegen (z. B. ../../data-bin)

    session_name:
        Name der Session, z. B. 'dev0001' (entspricht dem Ordnernamen).

    speaker_names:
        Liste der Sprecher-Namen für diese Session, z. B. ['SPK01', 'SPK02', 'SPK07'].

    Rückgabe:
        - D_session: Distanzmatrix (len(speaker_names) x len(speaker_names))
        - full_ids:  Liste der vollständigen IDs, z. B. ['dev0001_SPK01', ...]
    """
    ids, D = _load_ids_and_dist_matrix(data_dir, ids_file, dist_file)
    index_map = _build_index_map(ids)

    indices: List[int] = []
    full_ids: List[str] = []

    for spk in speaker_names:
        full_id = f"{session_name}_{spk}"
        if full_id not in index_map:
            raise KeyError(
                f"ID '{full_id}' nicht in ids.json gefunden. "
                f"Verfügbar sind z. B.: {ids[:5]} ..."
            )
        indices.append(index_map[full_id])
        full_ids.append(full_id)

    idx = np.array(indices, dtype=int)
    D_session = D[np.ix_(idx, idx)]

    return D_session, full_ids

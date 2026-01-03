# src/semantic/semantic_clustering.py
from __future__ import annotations

from typing import Dict, List
from sklearn.cluster import AgglomerativeClustering

from src.semantic.session_distances import get_session_semantic_distance_matrix


def semantic_cluster_speakers(
    data_dir: str,
    session_name: str,
    speaker_names: List[str],
    n_clusters: int | None = None,
    distance_threshold: float = 0.7,
    linkage: str = "complete",
) -> Dict[str, int]:
    """
    Rein semantisches Clustering auf Basis der session-spezifischen Distanzmatrix D_session.

    Erwartet:
      - data_dir enthält die semantischen Artefakte (z. B. ids.json, D.npy bzw. session-spezifische Ableitungen),
        so wie von src.semantic.session_distances.get_session_semantic_distance_matrix benötigt.

    Args:
        data_dir: Root-Verzeichnis der Semantik-Artefakte (z. B. .../_output/dev/semantik_clustering)
        session_name: z. B. "session_40"
        speaker_names: Liste der Sprecher-IDs dieser Session (z. B. ["spk_0", ...])
        n_clusters: Optional feste Clusteranzahl. Wenn None, wird distance_threshold genutzt.
        distance_threshold: Schwelle für AgglomerativeClustering im precomputed-distance space.
        linkage: Linkage-Methode (default: "complete"), konsistent zur Baseline-Logik.

    Returns:
        Mapping: speaker_id -> cluster_id
    """
    D_session, _full_ids = get_session_semantic_distance_matrix(
        data_dir=data_dir,
        session_name=session_name,
        speaker_names=speaker_names,
    )

    if n_clusters is None:
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=float(distance_threshold),
            metric="precomputed",
            linkage=linkage,
        )
    else:
        clustering = AgglomerativeClustering(
            n_clusters=int(n_clusters),
            metric="precomputed",
            linkage=linkage,
        )

    labels = clustering.fit_predict(D_session)

    return {spk: int(label) for spk, label in zip(speaker_names, labels)}

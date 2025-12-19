from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


# --------------------------------------------------------
# Projekt-Roots (analog zu seat_prior, aber separat)
# --------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

TEAM_C_ROOT = SCRIPT_DIR
while TEAM_C_ROOT.name != "team_c" and TEAM_C_ROOT.parent != TEAM_C_ROOT:
    TEAM_C_ROOT = TEAM_C_ROOT.parent

DATA_ROOT = TEAM_C_ROOT / "data-bin" / "dev"
OUTPUT_ROOT = DATA_ROOT / "output"


# --------------------------------------------------------
# Laden der Geometrie + ASD-Matching
# --------------------------------------------------------

def _load_seat_geometry(session_out_dir: Path) -> Tuple[List[int], np.ndarray]:
    """
    Lädt person_ids + dist_seat aus seat_geometry.npz.

    Erwartetes Format:
      - person_ids: int[N]
      - dist_seat : float[N,N], Werte 0..1 (normierte zirkulare Distanz)
    """
    npz_path = session_out_dir / "seat_geometry.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"seat_geometry.npz not found in {session_out_dir}")

    data = np.load(npz_path)
    person_ids = data["person_ids"].astype(int).tolist()
    dist_seat = np.asarray(data["dist_seat"], dtype=float)

    if dist_seat.shape[0] != len(person_ids) or dist_seat.shape[1] != len(person_ids):
        raise ValueError("dist_seat shape does not match person_ids length")

    return person_ids, dist_seat


def _load_asd_matching(session_out_dir: Path) -> Tuple[List[int], List[int], Dict[int, int], np.ndarray, List[int]]:
    """
    Lädt asd_seat_matching.json.

    Erwartete Struktur (aus deiner seat_prior-Pipeline):
      {
        "person_ids": [...],
        "asd_track_ids": [...],
        "angle_diff_deg_matrix": [[...], ...],  # Seats x ASD
        "assignments": [
          {"person_id": ..., "asd_track_id": ..., "angle_diff_deg": ...},
          ...
        ],
        "unassigned_asd_track_ids": [...]
      }

    Rückgabe:
      person_ids_match   : List[int]         (aus Matching-Datei)
      asd_track_ids      : List[int]
      person_to_asd      : {person_id: asd_id} für gematchte
      angle_diff_deg     : np.ndarray [N_seats, M_asd]
      unassigned_asd_ids : List[int]
    """
    path = session_out_dir / "asd_seat_matching.json"
    if not path.exists():
        raise FileNotFoundError(f"asd_seat_matching.json not found in {session_out_dir}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    person_ids_match = [int(p) for p in data.get("person_ids", [])]
    asd_track_ids = [int(a) for a in data.get("asd_track_ids", [])]
    angle_diff = np.asarray(data.get("angle_diff_deg_matrix", []), dtype=float)

    assignments = data.get("assignments", [])
    person_to_asd: Dict[int, int] = {}
    for a in assignments:
        try:
            pid = int(a["person_id"])
            sid = int(a["asd_track_id"])
        except (KeyError, TypeError, ValueError):
            continue
        person_to_asd[pid] = sid

    unassigned_asd_ids = [int(u) for u in data.get("unassigned_asd_track_ids", [])]

    # Sanity: shape
    if angle_diff.shape and (angle_diff.shape[0] != len(person_ids_match) or angle_diff.shape[1] != len(asd_track_ids)):
        raise ValueError("angle_diff_deg_matrix shape does not match person_ids/asd_track_ids lists")

    return person_ids_match, asd_track_ids, person_to_asd, angle_diff, unassigned_asd_ids


# --------------------------------------------------------
# Sitz-basierte Cluster (Personen-Ebene)
# --------------------------------------------------------

def _clusters_neighbors(person_ids: List[int]) -> Dict[int, int]:
    """
    Cluster-Variante 1:
      - Partition in Paare von Nachbarn entlang der Kreisreihenfolge.
      - Jeder Seat landet genau in einem Cluster.

    Beispiel [1,2,3,4,5]:
      -> {1:0, 2:0, 3:1, 4:1, 5:2}
    """
    n = len(person_ids)
    clusters: Dict[int, int] = {}
    used = set()
    cid = 0

    for idx, pid in enumerate(person_ids):
        if pid in used:
            continue

        nxt_idx = (idx + 1) % n
        pid_nxt = person_ids[nxt_idx]

        if pid_nxt in used or pid_nxt == pid:
            clusters[pid] = cid
            used.add(pid)
        else:
            clusters[pid] = cid
            clusters[pid_nxt] = cid
            used.add(pid)
            used.add(pid_nxt)

        cid += 1

    return clusters


def _clusters_opposites(person_ids: List[int]) -> Dict[int, int]:
    """
    Cluster-Variante 2:
      - Möglichst gegenüber liegende Personen in gemeinsame Cluster.
      - Für N gerade: saubere Paare mit Offset N/2.
      - Für N ungerade: Rest-Sitze ggf. alleine.
    """
    n = len(person_ids)
    clusters: Dict[int, int] = {}
    used = set()
    cid = 0

    half = n // 2

    for idx, pid in enumerate(person_ids):
        if pid in used:
            continue

        opp_idx = (idx + half) % n
        pid_opp = person_ids[opp_idx]

        if pid_opp in used or pid_opp == pid:
            clusters[pid] = cid
            used.add(pid)
        else:
            clusters[pid] = cid
            clusters[pid_opp] = cid
            used.add(pid)
            used.add(pid_opp)

        cid += 1

    return clusters


def _clusters_halves(person_ids: List[int]) -> Dict[int, int]:
    """
    Cluster-Variante 3:
      - Kreis in zwei Halbkreise schneiden.
      - Erster Halbkreis -> Cluster 0
      - Zweiter Halbkreis -> Cluster 1
    """
    n = len(person_ids)
    clusters: Dict[int, int] = {}

    split = (n + 1) // 2  # ceil
    first = person_ids[:split]
    second = person_ids[split:]

    for pid in first:
        clusters[pid] = 0
    for pid in second:
        clusters[pid] = 1 if second else 0

    return clusters


def _estimate_dist_threshold(dist_seat: np.ndarray) -> float:
    """
    Wählt einen heuristischen Schwellwert für "nah" vs. "weit" auf Basis dist_seat.

    Idee:
      - Für jeden Sitz: minimaler Abstand zu einem anderen Sitz.
      - median(min_abstände) = typischer Nachbarschafts-Abstand.
      - thr = 1.5 * median, aber geclamped in [0.15, 0.45].

    Dadurch passt sich thr an die Anzahl der Personen an und bleibt robust.
    """
    n = dist_seat.shape[0]
    if n <= 1:
        return 0.25

    mins = []
    for i in range(n):
        row = np.delete(dist_seat[i], i)
        if row.size:
            mins.append(float(np.min(row)))
    if not mins:
        return 0.25

    med = float(np.median(mins))
    thr = 1.5 * med
    thr = max(0.15, min(0.45, thr))
    return thr


def _clusters_dist_components(
    person_ids: List[int],
    dist_seat: np.ndarray,
    thr: float | None = None,
) -> Dict[int, int]:
    """
    Cluster-Variante 4:
      - Graph-Komponenten auf dist_seat:
        Knoten = Seats, Kante wenn dist_seat[i,j] <= thr.
      - thr adaptiv, wenn nicht vorgegeben.

    Das nutzt wirklich die Distanzmatrix und kann "Gruppen am Tisch" abbilden.
    """
    n = len(person_ids)
    if n == 0:
        return {}

    if thr is None:
        thr = _estimate_dist_threshold(dist_seat)

    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if dist_seat[i, j] <= thr:
                adj[i].append(j)
                adj[j].append(i)

    clusters: Dict[int, int] = {}
    visited = [False] * n
    cid = 0

    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        while stack:
            u = stack.pop()
            pid = person_ids[u]
            clusters[pid] = cid
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        cid += 1

    return clusters


def _clusters_dist_k(
    person_ids: List[int],
    dist_seat: np.ndarray,
    k: int = 2,
) -> Dict[int, int]:
    """
    Cluster-Variante 5:
      - Hierarchisches Clustering (average linkage) auf dist_seat.
      - Erzwingt k Cluster.

    Gut als "2-Gruppen-Baseline", z.B. zwei dominante Gesprächsgruppen.
    """
    n = len(person_ids)
    if n == 0:
        return {}
    if n == 1 or k <= 1:
        return {person_ids[0]: 0}

    condensed = squareform(dist_seat, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=k, criterion="maxclust").astype(int)

    uniq = sorted(set(labels.tolist()))
    remap = {old: new for new, old in enumerate(uniq)}

    clusters: Dict[int, int] = {}
    for i, pid in enumerate(person_ids):
        clusters[pid] = remap[int(labels[i])]
    return clusters


# --------------------------------------------------------
# Mapping: Personen-Cluster -> Speaker-Cluster
# --------------------------------------------------------

def _assign_speakers_from_person_clusters(
    person_ids: List[int],
    person_clusters: Dict[int, int],
    asd_track_ids: List[int],
    person_to_asd: Dict[int, int],
    angle_diff_deg: np.ndarray,
    unassigned_asd_ids: List[int],
) -> Dict[str, int]:
    """
    Kernlogik:
      - Bereits gematchte ASD-Speaker übernehmen den Cluster ihres Seats.
      - Unassigned ASD-Speaker werden anhand der Winkelmatrix dem
        "nächsten" Seat-Cluster zugeordnet.

    Input:
      person_ids      : Reihenfolge der Seats (Index 0..N-1)
      person_clusters : {person_id: cluster_id} (Person-Ebene)
      asd_track_ids   : Liste aller ASD-IDs, Reihenfolge = Spalten von angle_diff_deg
      person_to_asd   : {person_id: asd_id} für gematchte Speaker
      angle_diff_deg  : Matrix [N_seats, M_asd], Winkelabweichung (deg)
      unassigned_asd_ids: Liste der ASD-IDs ohne Seat-Mapping

    Output:
      { "spk_<id>": cluster_id }
    """
    # Map person_id -> Sitzindex
    idx_by_pid = {pid: i for i, pid in enumerate(person_ids)}

    # Map asd_id -> Spaltenindex in Winkelmatrix
    col_by_asd = {aid: j for j, aid in enumerate(asd_track_ids)}

    spk_clusters: Dict[str, int] = {}

    # 1) Gematchte ASD-Speaker: direkt vom Seat-Cluster übernehmen
    for pid, asd_id in person_to_asd.items():
        cid = person_clusters.get(pid)
        if cid is None:
            continue
        key = f"spk_{asd_id}"
        spk_clusters[key] = int(cid)

    # 2) Unassigned ASD-Speaker: anhand Winkelmatrix zum nächsten Seat-Cluster
    if angle_diff_deg.size > 0 and unassigned_asd_ids:
        # Winkel in "distanz-kompatibles" Maß normalisieren (0..1 ~ 0..180°)
        angle_dist = angle_diff_deg / 180.0

        for asd_id in unassigned_asd_ids:
            j = col_by_asd.get(asd_id)
            if j is None:
                # sollte nicht passieren, aber wir failen nicht hart
                continue

            col = angle_dist[:, j]  # Distanz von allen Seats zu diesem ASD
            if not np.isfinite(col).any():
                # komplett kaputt -> fallback: Cluster 0, wenn vorhanden
                if spk_clusters:
                    fallback_cid = sorted(set(spk_clusters.values()))[0]
                    spk_clusters[f"spk_{asd_id}"] = int(fallback_cid)
                else:
                    spk_clusters[f"spk_{asd_id}"] = 0
                continue

            # Index des Seats mit minimaler Distanz
            i_min = int(np.nanargmin(col))
            pid_nearest = person_ids[i_min]
            cid = person_clusters.get(pid_nearest)

            if cid is None:
                # kein Cluster für diesen Seat -> fallback: kleinstes existierendes Cluster
                if spk_clusters:
                    fallback_cid = sorted(set(spk_clusters.values()))[0]
                    cid = fallback_cid
                else:
                    cid = 0

            spk_clusters[f"spk_{asd_id}"] = int(cid)

    return spk_clusters


# --------------------------------------------------------
# Hauptfunktion pro Session
# --------------------------------------------------------

def build_seating_speaker_clusters_for_session(
    session_name: str,
    out_root: Path | None = None,
    make_neighbors: bool = True,
    make_opposites: bool = True,
    make_halves: bool = True,
    make_dist_components: bool = True,
    make_dist_k2: bool = True,
) -> Dict[str, Dict[str, int]]:
    """
    Baut mehrere Sitz-basierte Cluster-Heuristiken auf Speaker-Level und
    speichert sie als speaker_to_cluster_*.json im Session-Output-Ordner.

    Varianten:
      - neighbors       -> speaker_to_cluster_seat_neighbors.json
      - opposites       -> speaker_to_cluster_seat_opposites.json
      - halves          -> speaker_to_cluster_seat_halves.json
      - dist_components -> speaker_to_cluster_seat_dist_components.json
      - dist_k2         -> speaker_to_cluster_seat_dist_k2.json
    """
    if out_root is None:
        out_root = OUTPUT_ROOT

    session_out_dir = Path(out_root) / session_name
    if not session_out_dir.exists():
        raise FileNotFoundError(f"Session output dir not found: {session_out_dir}")

    # 1) Geometrie + Distanzmatrix (Seats)
    person_ids_geom, dist_seat = _load_seat_geometry(session_out_dir)

    # 2) ASD-Matching
    (
        person_ids_match,
        asd_track_ids,
        person_to_asd,
        angle_diff_deg,
        unassigned_asd_ids,
    ) = _load_asd_matching(session_out_dir)

    # Sicherstellen, dass Reihenfolge der person_ids konsistent ist
    if list(person_ids_geom) != list(person_ids_match):
        # Wenn die Reihenfolgen nicht identisch sind, mappen wir via person_id
        # und sortieren alles in die Geometrie-Reihenfolge um.
        idx_map = {pid: i for i, pid in enumerate(person_ids_match)}
        if angle_diff_deg.size:
            angle_diff_deg = np.stack(
                [angle_diff_deg[idx_map[pid]] for pid in person_ids_geom],
                axis=0,
            )

    person_ids = person_ids_geom  # definitive Reihenfolge

    results: Dict[str, Dict[str, int]] = {}

    # Helper: nimmt personen-Cluster und baut Speaker-Cluster + schreibt Datei
    def run_variant(tag: str, person_clusters: Dict[int, int]):
        if not person_clusters:
            return
        spk_clusters = _assign_speakers_from_person_clusters(
            person_ids=person_ids,
            person_clusters=person_clusters,
            asd_track_ids=asd_track_ids,
            person_to_asd=person_to_asd,
            angle_diff_deg=angle_diff_deg,
            unassigned_asd_ids=unassigned_asd_ids,
        )
        if not spk_clusters:
            return
        path = session_out_dir / f"speaker_to_cluster_seat_{tag}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(spk_clusters, f, indent=2)
        results[tag] = spk_clusters

    # 3a) neighbors
    if make_neighbors:
        pc = _clusters_neighbors(person_ids)
        run_variant("neighbors", pc)

    # 3b) opposites
    if make_opposites:
        pc = _clusters_opposites(person_ids)
        run_variant("opposites", pc)

    # 3c) halves
    if make_halves:
        pc = _clusters_halves(person_ids)
        run_variant("halves", pc)

    # 3d) distanz-basierte Komponenten (nutzt dist_seat voll aus)
    if make_dist_components:
        pc = _clusters_dist_components(person_ids, dist_seat, thr=None)
        run_variant("dist_components", pc)

    # 3e) distanz-basierte k=2-Cluster (hierarchisch)
    if make_dist_k2:
        pc = _clusters_dist_k(person_ids, dist_seat, k=2)
        run_variant("dist_k2", pc)

    return results


# --------------------------------------------------------
# Optional: alle Sessions durchlaufen
# --------------------------------------------------------

def main_all_sessions():
    """
    Einfacher Runner, um für alle Sessions unter OUTPUT_ROOT
    alle Sitz-basierten Speaker-Cluster-Heuristiken zu erzeugen.
    """
    for sdir in sorted(OUTPUT_ROOT.glob("session_*")):
        if not sdir.is_dir():
            continue
        session_name = sdir.name
        print(f"[seating_speaker_baselines] processing {session_name} ...")
        try:
            results = build_seating_speaker_clusters_for_session(session_name)
            variants = ", ".join(sorted(results.keys()))
            print(f"  -> variants: {variants}")
        except Exception as e:
            print(f"  ERROR in {session_name}: {e}")


if __name__ == "__main__":
    main_all_sessions()

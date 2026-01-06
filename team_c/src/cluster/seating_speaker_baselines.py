from __future__ import annotations

import json
import math
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

DATA_ROOT = TEAM_C_ROOT / "data-bin"
ASD_ROOT  = DATA_ROOT / "dev"
OUTPUT_ROOT = DATA_ROOT / "output_gaze"


# --------------------------------------------------------
# Laden der Geometrie + ASD-Matching
# --------------------------------------------------------

def _load_seat_geometry(session_out_dir: Path) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """
    Lädt person_ids + dist_seat + theta_deg aus seat_geometry.npz.

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

    theta_deg = data.get('theta_deg', None)
    if theta_deg is None:
        # Backward compatibility: if theta_deg is missing, approximate with evenly spaced angles
        n = len(person_ids)
        theta_deg = np.linspace(0.0, 360.0, num=max(n,1), endpoint=False, dtype=float)
    else:
        theta_deg = np.asarray(theta_deg, float)

    return person_ids, dist_seat, theta_deg


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
# Gaze + speaker distance matrices
# --------------------------------------------------------

# Speaker distance weights (combined = w_seat*seat + w_gaze*gaze)
W_SEAT_DEFAULT = 0.6
W_GAZE_DEFAULT = 0.4

# Interaction model for mutual gaze
# sigma in degrees: smaller => stricter "looking at"
GAZE_SIGMA_DEG = 25.0

# Clamp the required head turn (people won't look 170° through their head)
MAX_TARGET_TURN_DEG = 90.0

def _load_gaze_tracks(session_out_dir: Path) -> Dict[int, dict]:
    """Load gaze_tracks.json and return a dict {person_id: metrics}."""
    path = session_out_dir / "gaze_tracks.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        persons = data.get("persons", {}) or {}
        out: Dict[int, dict] = {}
        for k, v in persons.items():
            try:
                pid = int(k)
            except Exception:
                continue
            out[pid] = v
        return out
    except Exception:
        return {}

def _signed_circ_diff_deg(a: float, b: float) -> float:
    """Signed smallest difference b-a in degrees in (-180,180]."""
    d = (b - a + 180.0) % 360.0 - 180.0
    return float(d)

def _wrap180(x: float) -> float:
    return float((x + 180.0) % 360.0 - 180.0)

def _mutual_gaze_distance(
    theta_deg_i: float, yaw_i: float | None,
    theta_deg_j: float, yaw_j: float | None,
    yaw_sign_convention: float = 1.0,
) -> float:
    """Return a distance in [0,1] where 0 means strong mutual gaze evidence."""
    if yaw_i is None or yaw_j is None:
        return 1.0

    # Direction each person would need to turn (approx.) to look at the other.
    d_ij = _signed_circ_diff_deg(theta_deg_i, theta_deg_j) * float(yaw_sign_convention)
    d_ji = _signed_circ_diff_deg(theta_deg_j, theta_deg_i) * float(yaw_sign_convention)

    # Clamp unrealistic target turns
    t_ij = float(np.clip(d_ij, -MAX_TARGET_TURN_DEG, MAX_TARGET_TURN_DEG))
    t_ji = float(np.clip(d_ji, -MAX_TARGET_TURN_DEG, MAX_TARGET_TURN_DEG))

    e_ij = _wrap180(float(yaw_i) - t_ij)
    e_ji = _wrap180(float(yaw_j) - t_ji)

    # Gaussian affinity in [0,1]
    a_ij = math.exp(-(e_ij * e_ij) / (2.0 * GAZE_SIGMA_DEG * GAZE_SIGMA_DEG))
    a_ji = math.exp(-(e_ji * e_ji) / (2.0 * GAZE_SIGMA_DEG * GAZE_SIGMA_DEG))
    mutual = float(a_ij * a_ji)
    return float(np.clip(1.0 - mutual, 0.0, 1.0))

def build_speaker_distance_matrices_for_session(
    session_out_dir: Path,
    w_seat: float = W_SEAT_DEFAULT,
    w_gaze: float = W_GAZE_DEFAULT,
) -> dict | None:
    """Compute speaker↔speaker distance matrices (seat/gaze/combined) for one session.

    Strategy:
    - Speakers are taken ONLY from forced seat↔ASD assignments (extras dropped).
    - Seat distance is induced from dist_seat via the matched seat indices.
    - Gaze distance is interaction-based mutual gaze using head-yaw medians.
    """
    try:
        person_ids, dist_seat, theta_deg = _load_seat_geometry(session_out_dir)
        (person_ids_m, asd_track_ids, person_to_asd, angle_diff_deg) = _load_asd_matching(session_out_dir)
    except Exception:
        return None

    # Consistency: ensure geometry person_ids matches matching person_ids
    # If not, use geometry order and ignore mismatch by best-effort map.
    pid_to_idx = {int(pid): i for i, pid in enumerate(person_ids)}
    gaze = _load_gaze_tracks(session_out_dir)

    # Determine speakers from forced matches
    speaker_items = []
    for pid, sid in sorted(person_to_asd.items(), key=lambda x: int(x[1])):
        if int(pid) not in pid_to_idx:
            continue
        speaker_items.append((int(pid), int(sid)))

    if len(speaker_items) < 2:
        return {
            "speakers": [f"spk_{sid}" for _, sid in speaker_items],
            "matrices": {"seat": [], "gaze": [], "combined": []},
            "weights": {"seat": float(w_seat), "gaze": float(w_gaze)},
            "missing": {"gaze_speakers": [f"spk_{sid}" for _, sid in speaker_items]},
            "notes": {"reason": "not enough matched speakers"},
        }

    speakers = [f"spk_{sid}" for _, sid in speaker_items]
    seat_idx = [pid_to_idx[pid] for pid, _ in speaker_items]

    # Seat-induced speaker distance
    D_seat = dist_seat[np.ix_(seat_idx, seat_idx)].astype(float)

    # Yaw sign convention stored in gaze_tracks.json if present
    yaw_sign = 1.0
    try:
        gj = json.loads((session_out_dir / "gaze_tracks.json").read_text(encoding="utf-8"))
        yaw_sign = float(gj.get("yaw_sign_convention", 1.0))
    except Exception:
        pass

    # Gaze interaction distance
    yaws = []
    missing_gaze = []
    for (pid, sid) in speaker_items:
        m = gaze.get(pid, {})
        yaw = m.get("yaw_deg_median", None) if isinstance(m, dict) else None
        if yaw is None:
            yaws.append(None)
            missing_gaze.append(f"spk_{sid}")
        else:
            yaws.append(float(yaw))

    K = len(speakers)
    D_gaze = np.ones((K, K), float)
    for i in range(K):
        D_gaze[i, i] = 0.0
        for j in range(i + 1, K):
            th_i = float(theta_deg[seat_idx[i]])
            th_j = float(theta_deg[seat_idx[j]])
            d = _mutual_gaze_distance(th_i, yaws[i], th_j, yaws[j], yaw_sign_convention=yaw_sign)
            D_gaze[i, j] = D_gaze[j, i] = d

    # Combined distance: if gaze missing for a pair, fall back to seat-only weighting.
    w_seat = float(w_seat); w_gaze = float(w_gaze)
    D_comb = np.zeros_like(D_seat, float)
    for i in range(K):
        for j in range(K):
            if i == j:
                D_comb[i, j] = 0.0
                continue
            if yaws[i] is None or yaws[j] is None:
                # ignore gaze weight if not available
                D_comb[i, j] = D_seat[i, j]
            else:
                D_comb[i, j] = w_seat * D_seat[i, j] + w_gaze * D_gaze[i, j]

    return {
        "speakers": speakers,
        "matrices": {
            "seat": D_seat.tolist(),
            "gaze": D_gaze.tolist(),
            "combined": D_comb.tolist(),
        },
        "weights": {"seat": w_seat, "gaze": w_gaze},
        "missing": {"gaze_speakers": missing_gaze},
        "notes": {
            "speaker_source": "forced seat↔ASD assignments (extras dropped)",
            "gaze_interaction": "mutual gaze affinity from head-yaw medians (approx.)",
        },
    }

def _score_clustering(D: np.ndarray, labels: np.ndarray) -> float:
    # Higher is better: (between - within)
    n = D.shape[0]
    within = []
    between = []
    for i in range(n):
        for j in range(i+1, n):
            if labels[i] == labels[j]:
                within.append(D[i, j])
            else:
                between.append(D[i, j])
    if not within or not between:
        return -1e9
    return float(np.mean(between) - np.mean(within))

def choose_k_from_distance(D: np.ndarray, k_min: int = 2, k_max: int = 4) -> int:
    n = D.shape[0]
    if n < 2:
        return 1
    k_max = min(int(k_max), n)
    k_min = min(int(k_min), k_max)
    if k_min < 2:
        k_min = 2
    if k_max < 2:
        return 1

    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="average")
    best_k = k_min
    best_score = -1e18
    for k in range(k_min, k_max + 1):
        labels = fcluster(Z, t=k, criterion="maxclust")
        score = _score_clustering(D, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return int(best_k)

def agglomerative_labels_from_distance(D: np.ndarray, k: int) -> np.ndarray:
    if D.shape[0] == 0:
        return np.array([], int)
    if D.shape[0] == 1:
        return np.array([0], int)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=int(k), criterion="maxclust")
    # Normalize to 0..K-1
    uniq = {lab: i for i, lab in enumerate(sorted(set(labels.tolist())))}
    return np.array([uniq[x] for x in labels], int)


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
    person_ids_geom, dist_seat, theta_deg = _load_seat_geometry(session_out_dir)

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

    # ----------------------------------------------------
    # 4) Agglomerative clustering on speaker distance matrices
    # ----------------------------------------------------
    try:
        mats = build_speaker_distance_matrices_for_session(session_out_dir)
    except Exception:
        mats = None

    if mats is not None:
        # also store per-session matrices for debugging / reuse
        (session_out_dir / "speaker_distance_matrices.json").write_text(
            json.dumps(mats, indent=2), encoding="utf-8"
        )

        speakers = mats.get("speakers", [])
        matrices = mats.get("matrices", {}) or {}

        def write_agglom(tag: str, D_list):
            if not speakers or not D_list:
                return
            D = np.asarray(D_list, float)
            if D.shape[0] < 2:
                return
            k = choose_k_from_distance(D, k_min=2, k_max=4)
            labels = agglomerative_labels_from_distance(D, k=k)
            spk_clusters = {spk: int(labels[i]) for i, spk in enumerate(speakers)}
            out_path = session_out_dir / f"speaker_to_cluster_agglom_{tag}.json"
            out_path.write_text(json.dumps(spk_clusters, indent=2), encoding="utf-8")
            results[f"agglom_{tag}"] = spk_clusters

        write_agglom("seat", matrices.get("seat", []))
        write_agglom("gaze", matrices.get("gaze", []))
        write_agglom("combined", matrices.get("combined", []))

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

    try:
        p = write_global_speaker_distance_matrices_json()
        print(f"[seating_speaker_baselines] wrote global distance JSON: {p}")
    except Exception as e:
        print(f"[seating_speaker_baselines] ERROR writing global distance JSON: {e}")


def write_global_speaker_distance_matrices_json(out_path: Path | None = None) -> Path:
    """Build a single JSON containing all per-session speaker distance matrices."""
    if out_path is None:
        out_path = OUTPUT_ROOT / "speaker_distance_matrices.json"

    agg = {"version": 1, "sessions": {}}
    for sdir in sorted(OUTPUT_ROOT.glob("session_*")):
        if not sdir.is_dir():
            continue
        session_name = sdir.name
        per = None
        per_path = sdir / "speaker_distance_matrices.json"
        if per_path.exists():
            try:
                per = json.loads(per_path.read_text(encoding="utf-8"))
            except Exception:
                per = None
        if per is None:
            per = build_speaker_distance_matrices_for_session(sdir)
        if per is not None:
            agg["sessions"][session_name] = per

    out_path.write_text(json.dumps(agg, indent=2), encoding="utf-8")
    return out_path


if __name__ == "__main__":
    main_all_sessions()

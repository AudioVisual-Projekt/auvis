"""
Build a per-session seating prior from a 360° equirectangular central video
AND derive:
- a normalized circular seat–seat distance matrix
- visualizations of the seating geometry
- a mapping between seat IDs (video) and ASD speaker tracks (bbox.json)
- a dendrogram (hierarchical clustering) of seats
- a Ground-Truth-Cluster visualisation (which seats belong to which group)

Default input:
    team_c/data-bin/dev/dev_central_videos/session_*/central_video.mp4

Default seating output:
    team_c/data-bin/dev/output/session_*/{
        seat_order.json,
        adjacency.json,
        A.npy,
        metadata.json,
        summary.json,
        debug_labeled.jpg,

        seat_geometry.npz,
        seat_distance_heatmap.png,
        seating_geometry_polar.png,
        seat_dendrogram.png,

        asd_seat_matching.json,       (if ASD data present)
        asd_seat_matching.png,        (if ASD data present)
        seat_groundtruth_clusters.png (if labels + ASD-matching present)
    }

ASD bbox input (per session, default root = videos_root.parent = team_c/data-bin/dev):
    <asd_root>/session_*/speakers/spk_*/central_crops/track_00_bbox.json

Ground-truth Cluster input (per session, default root = videos_root.parent):
    <labels_root>/session_*/labels/speaker_to_cluster.json

Additionally appends one line per session to:
    team_c/data-bin/dev/output/sessions_index.jsonl

CLI flags (optional):
    --videos-root           override path to dev_central_videos
    --output-root           override path to output
    --asd-root              override ASD root (session_*/speakers/...)
                            default: videos_root.parent (team_c/data-bin/dev)
    --recompute-geometry    recompute seat_geometry.npz + PNGs even if they exist

Dependencies:
    pip install opencv-python mediapipe numpy scipy tqdm matplotlib
"""

from __future__ import annotations
import math


import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # reduce TF/TFLite verbosity early

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import cv2
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Try to quiet absl (used by mediapipe)
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass

# ----------------------------
# Config & default roots
# ----------------------------
from pathlib import Path

# ----------------------------
# Config & default roots
# ----------------------------
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Robust: gehe nach oben, bis wir im Ordner "team_c" landen
TEAM_C_ROOT = SCRIPT_DIR
while TEAM_C_ROOT.name != "team_c" and TEAM_C_ROOT.parent != TEAM_C_ROOT:
    TEAM_C_ROOT = TEAM_C_ROOT.parent

VIDEO_NAME = "central_video.mp4"

# Wir unterstützen zwei mögliche Layouts:
#   1) team_c/data-bin/dev/dev_central_videos   (aktuelles Setup)
#   2) team_c/data-bin/dev_central_videos       (altes Layout ohne "dev/")
candidate_data_roots = [
    TEAM_C_ROOT / "data-bin" / "dev",   # bevorzugt: .../data-bin/dev/...
    TEAM_C_ROOT / "data-bin",           # Fallback: .../data-bin/...
]

VIDEOS_ROOT = None
DATA_ROOT = None

for dr in candidate_data_roots:
    vr = dr / "dev_central_videos"
    if vr.exists():
        DATA_ROOT = dr
        VIDEOS_ROOT = vr
        break

# Falls nichts gefunden wurde: nimm das bevorzugte Layout als Default.
# (Der spätere Check in main() meldet dann "No such videos_root", falls es auch wirklich nicht existiert.)
if VIDEOS_ROOT is None:
    DATA_ROOT = candidate_data_roots[0]
    VIDEOS_ROOT = DATA_ROOT / "dev_central_videos"

# Output + ASD + Labels hängen am DATA_ROOT
OUTPUT_ROOT = DATA_ROOT / "output_gaze"
ASD_ROOT    = DATA_ROOT / "dev"      # dev/session_*/speakers/...
LABELS_ROOT = DATA_ROOT / "dev"      # dev/session_*/labels/...

# Standard: Geometrie immer frisch berechnen (kannst du bei Bedarf auf False setzen)
RECOMPUTE_GEOMETRY = True


# Sampling
SAMPLE_FPS = 4.0
WIN_STARTS = (0.0, 5.0, 15.0, 30.0)
WIN_MIN = 20.0
WIN_MAX = 45.0
MIN_FRAMES_WITH_DETS_TO_STOP = 5

# Detection tuning
MP_MIN_CONF = 0.35
MP_MODEL_SELECTION = 1
USE_HAAR_FALLBACK = True

# Track quality filters
MIN_TRACK_FRAC = 0.15
SIZE_PCTL = 30

# Tracking
IOU_MATCH_THRESH = 0.3
MAX_MISSES = 15

# Spatial prior
KAPPA = 4.0
MERGE_DEG = 15.0
LAMBDA_DEFAULT = 0.25

# ASD matching thresholds
ASSIGN_MAX_DEG = 35.0        # for per-frame box↔seat assignment (not used now)
ASD_MATCH_MAX_DEG = 45.0     # max allowed angle diff (deg) for Seat↔Speaker match

PAD_SCALE = 2.2          # brutal locker (1.8–2.5 sinnvoll)
MIN_FACE_SIDE = 160      # hoch für 360° Videos



# ----------------------------
# Face detection
# ----------------------------
try:
    import mediapipe as mp
    _MP_OK = True
except Exception:
    mp = None
    _MP_OK = False

def init_mp_fd(model_selection=MP_MODEL_SELECTION, min_conf=MP_MIN_CONF):
    if not _MP_OK:
        return None
    return mp.solutions.face_detection.FaceDetection(
        model_selection=model_selection,
        min_detection_confidence=min_conf
    )

def detect_faces_mediapipe(fd, frame_bgr):
    if fd is None:
        return []
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = fd.process(rgb)
    bboxes = []
    if res and res.detections:
        h, w = frame_bgr.shape[:2]
        for det in res.detections:
            rel = det.location_data.relative_bounding_box
            x1 = max(0, int(rel.xmin * w)); y1 = max(0, int(rel.ymin * h))
            x2 = min(w - 1, int((rel.xmin + rel.width) * w))
            y2 = min(h - 1, int((rel.ymin + rel.height) * h))
            if x2 > x1 and y2 > y1:
                bboxes.append([x1, y1, x2, y2])
    return bboxes

# ----------------------------
# Gaze / Head-Yaw estimation (optional)
# ----------------------------
# We estimate head yaw per tracked person. This is used downstream for
# interaction-based speaker distances ("mutual gaze" -> closer).
#
# Dependency note: This implementation prefers MediaPipe FaceMesh + solvePnP.
# If mediapipe is not installed or FaceMesh fails, we fall back to "no gaze"
# (n_samples=0) without failing the pipeline.
#
# Sign convention:
#   yaw_deg > 0  => looking towards increasing seat theta (counter-clockwise)
# This is an approximation because yaw is measured in camera/image coordinates.
# If you find the sign is flipped in your data, set YAW_SIGN_CONVENTION = -1.0.
YAW_SIGN_CONVENTION = 1.0

# How often to run yaw estimation for a track when it is updated (every sampled frame by default).
GAZE_EVERY_N_UPDATES = 1

# FaceMesh parameters (kept lightweight)
FACEMESH_STATIC_IMAGE_MODE = False
FACEMESH_MAX_NUM_FACES = 1
FACEMESH_REFINE_LANDMARKS = False
FACEMESH_MIN_DET_CONF = 0.2
FACEMESH_MIN_TRACK_CONF = 0.2

# Landmark indices (MediaPipe FaceMesh)
# Commonly used points for head pose (approximate):
#  - Nose tip: 1
#  - Chin: 152
#  - Left eye outer corner: 33
#  - Right eye outer corner: 263
#  - Left mouth corner: 61
#  - Right mouth corner: 291
POSE_LANDMARKS = {
    "nose": 1,
    "chin": 152,
    "left_eye": 33,
    "right_eye": 263,
    "left_mouth": 61,
    "right_mouth": 291,
}

_POSE_LM = POSE_LANDMARKS


def init_mp_facemesh():
    if not _MP_OK:
        return None
    try:
        return mp.solutions.face_mesh.FaceMesh(
            static_image_mode=FACEMESH_STATIC_IMAGE_MODE,
            max_num_faces=FACEMESH_MAX_NUM_FACES,
            refine_landmarks=FACEMESH_REFINE_LANDMARKS,
            min_detection_confidence=FACEMESH_MIN_DET_CONF,
            min_tracking_confidence=FACEMESH_MIN_TRACK_CONF,
        )
    except Exception:
        return None

def _extract_pose_points(face_landmarks, w: int, h: int):
    # returns 2D image points for PnP, in pixel coords
    pts = []
    for k in ("nose","chin","left_eye","right_eye","left_mouth","right_mouth"):
        idx = POSE_LANDMARKS[k]
        lm = face_landmarks.landmark[idx]
        pts.append([lm.x * w, lm.y * h])
    return np.asarray(pts, dtype=np.float64)

def _get_3d_model_points():
    # Generic 3D model points in arbitrary units.
    # These approximate the relative layout of the facial features.
    return np.array([
        [0.0, 0.0, 0.0],        # nose tip
        [0.0, -63.6, -12.5],    # chin
        [-43.3, 32.7, -26.0],   # left eye corner
        [43.3, 32.7, -26.0],    # right eye corner
        [-28.9, -28.9, -24.1],  # left mouth corner
        [28.9, -28.9, -24.1],   # right mouth corner
    ], dtype=np.float64)

_MODEL_POINTS_3D = _get_3d_model_points()

def estimate_head_yaw_deg_from_crop(face_bgr: np.ndarray, fm) -> float | None:
    """
    Estimate head yaw (degrees) from a face crop (BGR). Returns None on failure.

    IMPORTANT: In 360° equirectangular central videos, a full 3D head-pose solvePnP often
    fails (distortion + tiny faces). Therefore we:
      1) try FaceMesh + solvePnP (if it works)
      2) otherwise fall back to a robust 2D landmark yaw proxy:
         yaw_proxy ~ (nose_x - mid_eye_x) / eye_distance  -> degrees
    This proxy is sufficient for the downstream "mutual gaze" interaction signal.
    """
    if fm is None or face_bgr is None or face_bgr.size == 0:
        return None
    try:
        img_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        res = fm.process(img_rgb)
        if not res or not res.multi_face_landmarks:
            return None

        h, w = face_bgr.shape[:2]
        lm = res.multi_face_landmarks[0]

        # --- robust 2D yaw proxy (works even when solvePnP fails) ---
        try:
            # Use FaceMesh landmarks: nose tip (1), left eye outer (33), right eye outer (263)
            nose = lm.landmark[_POSE_LM["nose"]]
            le   = lm.landmark[_POSE_LM["left_eye"]]
            re_  = lm.landmark[_POSE_LM["right_eye"]]
            nose_x = nose.x * w
            le_x   = le.x * w
            re_x   = re_.x * w
            mid_eye_x = 0.5 * (le_x + re_x)
            eye_dist = max(1.0, abs(re_x - le_x))

            # Normalize nose offset by eye distance.
            rel = (nose_x - mid_eye_x) / eye_dist  # roughly in [-0.6, 0.6] for typical yaws

            # Map to degrees. Empirically, rel≈0.35 corresponds to ~30-40° yaw on many crops.
            yaw_proxy_deg = float(np.clip(rel * 90.0, -90.0, 90.0))  # conservative
        except Exception:
            yaw_proxy_deg = None

        # --- attempt solvePnP pose yaw (optional) ---
        try:
            image_points = _extract_pose_points(lm, w, h)

            focal = float(w)
            cam = np.array([[focal, 0, w/2.0],
                            [0, focal, h/2.0],
                            [0, 0, 1]], dtype=np.float64)
            dist = np.zeros((4, 1), dtype=np.float64)

            ok, rvec, tvec = cv2.solvePnP(
                _MODEL_POINTS_3D, image_points, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if ok:
                rmat, _ = cv2.Rodrigues(rvec)
                sy = math.sqrt(rmat[0, 0]*rmat[0, 0] + rmat[1, 0]*rmat[1, 0])
                singular = sy < 1e-6
                if not singular:
                    yaw = math.atan2(rmat[2, 0], sy)
                else:
                    yaw = math.atan2(-rmat[1, 2], rmat[1, 1])

                yaw_deg = float(np.degrees(yaw))
                # If yaw is crazy due to failed pose, ignore and use proxy
                if -120.0 <= yaw_deg <= 120.0:
                    return yaw_deg
        except Exception:
            pass

        return yaw_proxy_deg
    except Exception:
        return None

# Fallback: Haar
_HAAR = None
def _get_haar():
    global _HAAR
    if _HAAR is None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        haar = cv2.CascadeClassifier(path)
        _HAAR = haar if not haar.empty() else None
    return _HAAR

def detect_faces_fallback_haar(frame_bgr):
    haar = _get_haar()
    if haar is None:
        return []
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=5, minSize=(48, 48))
    bboxes = []
    for (x, y, w, h) in faces:
        bboxes.append([int(x), int(y), int(x + w), int(y + h)])
    return bboxes

# ----------------------------
# Tracking
# ----------------------------
def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / (ua + 1e-9)

class Track:
    def __init__(self, tid: int, bbox, frame_idx: int):
        self.id = tid
        self.bbox = bbox
        self.last_seen = frame_idx
        self.x_centers: List[float] = []
        self.y_centers: List[float] = []
        self.heights: List[float] = []
        self.update_bbox(bbox)

    def update_bbox(self, bbox):
        x1, y1, x2, y2 = bbox
        self.bbox = bbox
        self.x_centers.append(0.5 * (x1 + x2))
        self.y_centers.append(0.5 * (y1 + y2))
        self.heights.append(y2 - y1)

def update_tracks(active: Dict[int, Track], detections: List[List[int]], frame_idx: int, next_id: int) -> int:
    if not detections and not active:
        return next_id
    if not active:
        for d in detections:
            active[next_id] = Track(next_id, d, frame_idx)
            next_id += 1
        return next_id

    tids = list(active.keys())
    A = np.zeros((len(tids), len(detections)), dtype=float)
    for i, tid in enumerate(tids):
        for j, det in enumerate(detections):
            A[i, j] = 1.0 - iou(active[tid].bbox, det)

    if A.size:
        ri, cj = linear_sum_assignment(A)
    else:
        ri, cj = np.array([], int), np.array([], int)

    matched_t = set()
    matched_d = set()
    for i, j in zip(ri, cj):
        if 1.0 - A[i, j] >= IOU_MATCH_THRESH:
            tid = tids[i]
            active[tid].update_bbox(detections[j])
            active[tid].last_seen = frame_idx
            matched_t.add(tid)
            matched_d.add(j)

    for j, det in enumerate(detections):
        if j not in matched_d:
            active[next_id] = Track(next_id, det, frame_idx)
            next_id += 1

    to_del = [tid for tid, tr in active.items() if (frame_idx - tr.last_seen) > MAX_MISSES]
    for tid in to_del:
        del active[tid]
    return next_id

# ----------------------------
# Geometry helpers
# ----------------------------
def crop_letterbox(frame):
    h, w = frame.shape[:2]
    content_h = int(w / 2)
    if content_h >= h:
        return frame, (0, h)
    y1 = (h - content_h) // 2
    y2 = y1 + content_h
    return frame[y1:y2, :], (y1, y2)

def smallest_circ_dist_deg(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)

def pick_seam_largest_gap(thetas_deg: np.ndarray) -> float:
    th = np.sort(thetas_deg % 360.0)
    diffs = np.diff(np.r_[th, th[0] + 360.0])
    i = int(np.argmax(diffs))
    return float((th[i] + diffs[i] / 2.0) % 360.0)

@dataclass
class SeatOrderItem:
    person_id: int
    track_id: int
    theta_deg: float
    neighbors: Tuple[int, int]

def compute_seat_order_and_A(track_x: Dict[int, List[float]], frame_width: int,
                             kappa: float = KAPPA, merge_deg: float = MERGE_DEG
                             ) -> Tuple[List[SeatOrderItem], np.ndarray, float]:
    tracks = []
    for tid, xs in track_x.items():
        if xs:
            theta = (360.0 * float(np.median(xs)) / float(frame_width)) % 360.0
            tracks.append((tid, theta))
    if not tracks:
        return [], np.zeros((0, 0), float), 0.0

    raw = np.array([t[1] for t in tracks], float)
    seam = pick_seam_largest_gap(raw)
    shifted = [(tid, (th - seam) % 360.0) for tid, th in tracks]
    shifted.sort(key=lambda x: x[1])

    dedup = []
    for tid, th in shifted:
        if dedup and smallest_circ_dist_deg(th, dedup[-1][1]) <= merge_deg:
            continue
        dedup.append((tid, th))

    N = len(dedup)
    seat_order = []
    for i, (tid, th) in enumerate(dedup):
        prev_id = ((i - 1) % N) + 1
        next_id = ((i + 1) % N) + 1
        seat_order.append(SeatOrderItem(person_id=i + 1, track_id=tid, theta_deg=float(th),
                                        neighbors=(prev_id, next_id)))

    theta = np.deg2rad([s.theta_deg for s in seat_order])
    A = np.zeros((N, N), float)
    for i in range(N):
        for j in range(N):
            d = np.arccos(np.cos(theta[i] - theta[j]))
            A[i, j] = np.exp(kappa * np.cos(d))
    if A.size:
        A /= max(A.max(), 1e-12)
    return seat_order, A, seam

# ----------------------------
# Seat geometry NPZ + Plots + Dendrogram
# ----------------------------
def ensure_seat_geometry_and_plots(out_dir: Path,
                                   seat_order: List[SeatOrderItem],
                                   seam_deg: float,
                                   frame_width: int,
                                   recompute: bool = False) -> dict:
    npz_path = out_dir / "seat_geometry.npz"
    heatmap_path = out_dir / "seat_distance_heatmap.png"
    polar_path = out_dir / "seating_geometry_polar.png"
    dendro_path = out_dir / "seat_dendrogram.png"

    if npz_path.exists() and not recompute:
        data = np.load(npz_path)
        dist_seat = data["dist_seat"]
        theta_deg = data["theta_deg"]
        person_ids = data["person_ids"]
        track_ids = data["track_ids"]
        seam_saved = float(data["seam_deg"])
        if not heatmap_path.exists():
            _plot_seat_heatmap(heatmap_path, dist_seat, person_ids)
        if not polar_path.exists():
            _plot_seating_polar(polar_path, theta_deg, person_ids, seam_saved)
        if not dendro_path.exists():
            _plot_seat_dendrogram(dendro_path, dist_seat, person_ids)
        return {
            "dist_seat": dist_seat,
            "theta_deg": theta_deg,
            "person_ids": person_ids,
            "track_ids": track_ids,
            "seam_deg": seam_saved,
        }

    theta_deg = np.array([s.theta_deg for s in seat_order], float)
    person_ids = np.array([s.person_id for s in seat_order], int)
    track_ids = np.array([s.track_id for s in seat_order], int)

    N = len(theta_deg)
    dist_seat = np.zeros((N, N), float)
    for i in range(N):
        for j in range(N):
            dist_seat[i, j] = smallest_circ_dist_deg(theta_deg[i], theta_deg[j]) / 180.0

    np.savez(
        npz_path,
        dist_seat=dist_seat,
        theta_deg=theta_deg,
        person_ids=person_ids,
        track_ids=track_ids,
        seam_deg=float(seam_deg),
    )

    _plot_seat_heatmap(heatmap_path, dist_seat, person_ids)
    _plot_seating_polar(polar_path, theta_deg, person_ids, seam_deg)
    _plot_seat_dendrogram(dendro_path, dist_seat, person_ids)

    return {
        "dist_seat": dist_seat,
        "theta_deg": theta_deg,
        "person_ids": person_ids,
        "track_ids": track_ids,
        "seam_deg": float(seam_deg),
    }

def _plot_seat_heatmap(path: Path, dist_seat: np.ndarray, person_ids: np.ndarray):
    N = dist_seat.shape[0]
    if N == 0:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(dist_seat, origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(person_ids)
    ax.set_yticklabels(person_ids)
    ax.set_xlabel("Person ID")
    ax.set_ylabel("Person ID")
    ax.set_title("Seat distance matrix (0 = same, 1 = opposite)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("normalized circular distance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def _plot_seating_polar(path: Path, theta_deg: np.ndarray, person_ids: np.ndarray, seam_deg: float):
    if len(theta_deg) == 0:
        return
    theta_global = (theta_deg + seam_deg) % 360.0
    theta_rad = np.deg2rad(theta_global)
    r = np.ones_like(theta_rad)

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"})
    ax.scatter(theta_rad, r)
    for pid, th, rr in zip(person_ids, theta_rad, r):
        ax.text(th, rr + 0.05, str(int(pid)), ha="center", va="center")

    ax.set_title(f"Seating geometry (N={len(person_ids)}, seam={seam_deg:.1f}°)")
    ax.set_rmax(1.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def _plot_seat_dendrogram(path: Path, dist_seat: np.ndarray, person_ids: np.ndarray):
    N = dist_seat.shape[0]
    if N <= 1:
        return
    condensed = squareform(dist_seat, checks=False)
    Z = linkage(condensed, method="average")
    fig, ax = plt.subplots(figsize=(6, 4))
    dendrogram(Z, labels=[str(int(pid)) for pid in person_ids], ax=ax)
    ax.set_xlabel("Person ID")
    ax.set_ylabel("dissimilarity (normalized circular distance)")
    ax.set_title("Hierarchical clustering of seats")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

# ----------------------------
# ASD bbox -> Winkel -> seat matching
# ----------------------------
def load_asd_track_angles(asd_session_dir: Path,
                          frame_width: int,
                          seam_deg: float) -> Dict[int, float]:
    """
    ASD structure per session (default root = videos_root.parent):

        asd_session_dir/
            speakers/
                spk_0/central_crops/track_00_bbox.json
                spk_1/central_crops/track_00_bbox.json
                ...

    Returns:
        {speaker_id (int): theta_deg_seam_relative (float)}
    """
    angles: Dict[int, float] = {}

    speakers_root = asd_session_dir / "speakers"
    if not speakers_root.exists():
        return angles

    for spk_dir in sorted(speakers_root.glob("spk_*")):
        try:
            speaker_id = int(spk_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        bbox_path = spk_dir / "central_crops" / "track_00_bbox.json"
        if not bbox_path.exists():
            continue

        try:
            with bbox_path.open("r", encoding="utf-8") as f:
                dets = json.load(f)
        except Exception:
            continue

        xs = []
        for _, box in dets.items():
            try:
                x1 = float(box["x1"])
                x2 = float(box["x2"])
            except (KeyError, TypeError, ValueError):
                continue
            xs.append(0.5 * (x1 + x2))

        if not xs:
            continue

        xc_med = float(np.median(xs))
        theta_raw = 360.0 * xc_med / float(frame_width)
        theta_shift = (theta_raw - seam_deg) % 360.0
        angles[speaker_id] = theta_shift

    return angles

def match_asd_tracks_and_plot(geom: dict,
                              asd_session_dir: Path,
                              frame_width: int,
                              out_dir: Path) -> dict:
    """
    Erzwinge ein 1:1 Matching zwischen Sitzplätzen und einem Subset der ASD-Speaker.
    - Wenn es mehr ASD-Speaker als Seats gibt: nur so viele ASD-Speaker wie Seats verwenden,
      der Rest bleibt ungematcht.
    - Wenn es mehr Seats als ASD-Speaker gibt: einige Seats bleiben ohne ASD-ID.

    Das Matching wird immer eingetragen, unabhängig vom Winkelabstand.
    """
    if not asd_session_dir.exists():
        return {"N_asd_tracks": 0, "N_asd_used": 0, "N_pairs": 0,
                "assignments": [], "asd_track_ids": [], "unassigned_asd_track_ids": []}
    print(f"[asd-debug] asd_session_dir={asd_session_dir} exists={asd_session_dir.exists()}")
    speakers_root = asd_session_dir / "speakers"
    print(f"[asd-debug] speakers_root={speakers_root} exists={speakers_root.exists()}")


    asd_angles = load_asd_track_angles(asd_session_dir, frame_width, geom["seam_deg"])
    print(f"[asd-debug] loaded_asd_angles={len(asd_angles)}")
    if not asd_angles:
        out_json = {
            "person_ids": list(map(int, geom.get("person_ids", []))),
            "asd_track_ids": [],
            "angle_diff_deg_matrix": [],
            "assignments": [],
            "assignments_all": [],
            "angle_diff_max_deg_for_match": float(ASD_MATCH_MAX_DEG),
            "unassigned_asd_track_ids": [],
            "note": "No ASD tracks found/loaded for this session.",
        }
        (out_dir / "asd_seat_matching.json").write_text(json.dumps(out_json, indent=2), encoding="utf-8")
        return {"N_asd_tracks": 0, "N_asd_used": 0, "N_pairs": 0,
                "assignments": [], "asd_track_ids": [], "unassigned_asd_track_ids": []}

    seat_theta = np.asarray(geom["theta_deg"], float)
    person_ids = np.asarray(geom["person_ids"], int)

    asd_track_ids = np.array(sorted(asd_angles.keys()), int)
    theta_asd = np.array([asd_angles[tid] for tid in asd_track_ids], float)

    N = len(seat_theta)
    M = len(theta_asd)
    if N == 0 or M == 0:
        return {"N_asd_tracks": int(M), "N_asd_used": 0, "N_pairs": 0,
                "assignments": [], "asd_track_ids": asd_track_ids.tolist(),
                "unassigned_asd_track_ids": asd_track_ids.tolist()}

    # Kostenmatrix: kleiner = besser
    C = np.zeros((N, M), float)
    for i in range(N):
        for j in range(M):
            C[i, j] = smallest_circ_dist_deg(seat_theta[i], theta_asd[j])

    # Hungarian: erzeugt immer min(N, M) Zuordnungen
    ri, cj = linear_sum_assignment(C)

    assignments = []
    assigned_asd_ids = set()
    for i, j in zip(ri, cj):
        d = float(C[i, j])
        pid = int(person_ids[i])
        sid = int(asd_track_ids[j])
        assignments.append({
            "person_id": pid,
            "asd_track_id": sid,
            "angle_diff_deg": d,          # für Task 5 gut sichtbar lassen
            "is_confident": bool(d <= ASD_MATCH_MAX_DEG),
            "method": "hungarian_1to1",
        })
        assigned_asd_ids.add(sid)

    # per-speaker argmin (many-to-one allowed): ensures every ASD speaker can be mapped to a seat
    assignments_all = []
    for j in range(M):
        i_best = int(np.argmin(C[:, j]))
        d_best = float(C[i_best, j])
        assignments_all.append({
            "person_id": int(person_ids[i_best]),
            "asd_track_id": int(asd_track_ids[j]),
            "angle_diff_deg": d_best,
            "is_confident": bool(d_best <= ASD_MATCH_MAX_DEG),
            "method": "per_speaker_argmin",
        })

    unassigned_asd = sorted([int(tid) for tid in asd_track_ids if int(tid) not in assigned_asd_ids])

    out_json = {
        "person_ids": person_ids.tolist(),
        "asd_track_ids": asd_track_ids.tolist(),
        "angle_diff_deg_matrix": C.tolist(),
        "assignments": assignments,
        "assignments_all": assignments_all,
        # Schwelle nur noch als Info, wird nicht mehr zum Filtern verwendet:
        "angle_diff_max_deg_for_match": float(ASD_MATCH_MAX_DEG),
        "unassigned_asd_track_ids": unassigned_asd,
    }
    (out_dir / "asd_seat_matching.json").write_text(
        json.dumps(out_json, indent=2), encoding="utf-8"
    )

    # Heatmap + markierte Matches
    C_norm = C / 180.0
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(C_norm, origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(M))
    ax.set_xticklabels(asd_track_ids)
    ax.set_yticks(range(N))
    ax.set_yticklabels(person_ids)
    ax.set_xlabel("ASD speaker ID (spk_X)")
    ax.set_ylabel("Seat person ID")
    ax.set_title("Seat ↔ ASD-track angle distance\n(white circles = forced matches)")

    for i, j in zip(ri, cj):
        ax.scatter(j, i, facecolors="none", edgecolors="white", linewidths=1.8, s=70)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("normalized circular distance")
    fig.tight_layout()
    fig.savefig(out_dir / "asd_seat_matching.png", dpi=150)
    plt.close(fig)

    return {
        "N_asd_tracks": int(M),
        "N_asd_used": int(len(assigned_asd_ids)),
        "N_pairs": int(len(assignments)),
        "assignments": assignments,
        "assignments_all": assignments_all,
        "asd_track_ids": asd_track_ids.tolist(),
        "unassigned_asd_track_ids": unassigned_asd,
    }


# ----------------------------
# Ground-truth Cluster
# ----------------------------
def load_speaker_clusters(session_name: str) -> Dict[int, int]:
    """
    Load speaker_to_cluster.json:

        LABELS_ROOT/session_XX/labels/speaker_to_cluster.json

    Format:
        { "spk_0": 0, "spk_1": 0, "spk_2": 1, ... }

    Returns:
        { speaker_id(int): cluster_id(int) }
    """
    if LABELS_ROOT is None:
        return {}
    labels_file = LABELS_ROOT / session_name / "labels" / "speaker_to_cluster.json"
    if not labels_file.exists():
        return {}

    try:
        with labels_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}

    spk_to_cluster: Dict[int, int] = {}
    for k, v in data.items():
        if not isinstance(k, str) or not k.startswith("spk_"):
            continue
        try:
            spk_id = int(k.split("_")[1])
            cluster_id = int(v)
        except (IndexError, ValueError, TypeError):
            continue
        spk_to_cluster[spk_id] = cluster_id
    return spk_to_cluster

def _plot_groundtruth_clusters(path: Path,
                               geom: dict,
                               assignments: List[dict],
                               spk_to_cluster: Dict[int, int]):
    """
    Draw a polar plot of the seating layout,
    color-coded by ground-truth conversation clusters.

    Mapping:
        person_id --(ASD-matching)--> speaker_id (spk_X)
                   --(Labels)--> cluster_id
    """
    theta_deg = np.asarray(geom["theta_deg"], float)
    person_ids = np.asarray(geom["person_ids"], int)
    seam_deg = float(geom["seam_deg"])

    if theta_deg.size == 0 or not assignments or not spk_to_cluster:
        return

    pid_to_cluster: Dict[int, int] = {}
    for a in assignments:
        pid = int(a["person_id"])
        spk_id = int(a["asd_track_id"])
        if spk_id in spk_to_cluster:
            pid_to_cluster[pid] = spk_to_cluster[spk_id]

    if not pid_to_cluster:
        return

    theta_global = (theta_deg + seam_deg) % 360.0
    theta_rad = np.deg2rad(theta_global)
    r = np.ones_like(theta_rad)

    unique_clusters = sorted(set(pid_to_cluster.values()))
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(unique_clusters))))
    cluster_to_color = {cid: colors[i] for i, cid in enumerate(unique_clusters)}

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection": "polar"})

    # nur Seats mit Cluster berücksichtigen
    idx_all = [i for i, pid in enumerate(person_ids) if pid in pid_to_cluster]
    if not idx_all:
        return

    # dann wie gehabt, aber nur über idx_all / pid_to_cluster gehen
    for cid in unique_clusters:
        idx = [i for i in idx_all if pid_to_cluster.get(int(person_ids[i])) == cid]
        if not idx:
            continue
        ax.scatter(theta_rad[idx], r[idx], color=cluster_to_color[cid], s=80,
                   label=f"cluster {cid}", zorder=2)

    # Labels nur für Seats mit Cluster
    for i in idx_all:
        pid = int(person_ids[i])
        th = theta_rad[i]
        rr = r[i]
        label = f"{pid} (c{pid_to_cluster[pid]})"
        ax.text(th, rr + 0.07, label, ha="center", va="center", fontsize=8)

    # labels: Person ID (+ cluster)
    for pid, th, rr in zip(person_ids, theta_rad, r):
        label = str(int(pid))
        ax.text(th, rr + 0.07, label, ha="center", va="center", fontsize=8)

    ax.set_title("Ground-truth conversation clusters (seat layout)")
    ax.set_rmax(1.3)
    ax.set_rticks([])  # entfernt radiale Tick-Positionen
    ax.set_yticklabels([])  # falls Matplotlib dennoch Labels setzen will
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

# ----------------------------
# Main per-session processing
# ----------------------------
def process_video(session_dir: Path):
    gaze_calls = 0
    gaze_hits = 0

    video = session_dir / VIDEO_NAME
    if not video.exists():
        return False, {"session": session_dir.name, "error": "missing video"}

    out_dir = (OUTPUT_ROOT / session_dir.name)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return False, {"session": session_dir.name, "error": "cannot open"}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / max(fps, 1e-6)

    win_starts = set(WIN_STARTS)
    win_starts.add(max(0.0, duration * 0.5 - 10.0))
    win_starts = sorted(win_starts)
    window_duration = float(np.clip(duration * 0.5, WIN_MIN, WIN_MAX))

    content0 = None
    y_band = (0, 0)
    w_c = h_c = None

    active: Dict[int, Track] = {}
    next_id = 1
    track_x: Dict[int, List[float]] = {}
    track_h: Dict[int, List[float]] = {}
    track_y: Dict[int, List[float]] = {}
    track_yaw: Dict[int, List[float]] = {}   # head yaw estimates per track_id
    track_updates: Dict[int, int] = {}       # how many times a track was updated (for subsampling yaw estimation)
    frames_with_dets = 0

    fm = init_mp_facemesh()

    for win_start in win_starts:
        start = min(win_start, max(0.0, duration - 1.0))
        end = min(duration, start + window_duration)
        step = max(int(round(fps / SAMPLE_FPS)), 1)
        start_f = int(round(start * fps))
        end_f = int(round(end * fps))

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        ok, frame0 = cap.read()
        if not ok:
            continue
        content0_win, y_band_win = crop_letterbox(frame0)
        h_c_win, w_c_win = content0_win.shape[:2]
        if content0 is None:
            content0, y_band = content0_win.copy(), y_band_win
            h_c, w_c = h_c_win, w_c_win

        fd = init_mp_fd(model_selection=MP_MODEL_SELECTION, min_conf=MP_MIN_CONF)

        pbar = tqdm(total=max(0, end_f - start_f), desc=f"{session_dir.name}@{int(start)}s")
        f = start_f
        while f < end_f:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok:
                break
            content, _ = crop_letterbox(frame)

            bboxes = detect_faces_mediapipe(fd, content)
            if not bboxes and USE_HAAR_FALLBACK:
                bboxes = detect_faces_fallback_haar(content)

            if bboxes:
                frames_with_dets += 1

            next_id = update_tracks(active, bboxes, frame_idx=f, next_id=next_id)

            for tid, tr in active.items():
                if tr.last_seen == f and tr.x_centers:
                    track_x.setdefault(tid, []).append(tr.x_centers[-1])
                    track_y.setdefault(tid, []).append(tr.y_centers[-1])
                    track_h.setdefault(tid, []).append(tr.heights[-1])

                    # Optional head-yaw estimation for gaze interactions
                    track_updates[tid] = track_updates.get(tid, 0) + 1
                    if (track_updates[tid] % GAZE_EVERY_N_UPDATES) == 0:
                        try:
                            x1, y1, x2, y2 = map(int, tr.bbox)

                            # bbox center + scaled padding
                            cx = 0.5 * (x1 + x2)
                            cy = 0.5 * (y1 + y2)
                            bw = max(1, x2 - x1)
                            bh = max(1, y2 - y1)

                            half_w = 0.5 * bw * PAD_SCALE
                            half_h = 0.5 * bh * PAD_SCALE

                            x1p = int(max(0, cx - half_w))
                            x2p = int(min(content.shape[1] - 1, cx + half_w))
                            y1p = int(max(0, cy - half_h))
                            y2p = int(min(content.shape[0] - 1, cy + half_h))

                            crop = content[y1p:y2p, x1p:x2p]

                            # If crop is tiny, upscale for FaceMesh stability.
                            if crop is not None and crop.size > 0:
                                ch, cw = crop.shape[:2]
                                min_side = min(ch, cw)
                                if min_side < MIN_FACE_SIDE:
                                    scale = MIN_FACE_SIDE / float(min_side)
                                    crop = cv2.resize(
                                        crop,
                                        (int(cw * scale), int(ch * scale)),
                                        interpolation=cv2.INTER_CUBIC
                                    )

                            yaw = estimate_head_yaw_deg_from_crop(crop, fm)
                            if yaw is not None:
                                track_yaw.setdefault(tid, []).append(float(yaw))
                        except Exception:
                            pass

            f += step
            pbar.update(step)
        pbar.close()

        if fd is not None:
            try:
                fd.close()
            except Exception:
                pass

        if frames_with_dets >= MIN_FRAMES_WITH_DETS_TO_STOP:
            break

    if frames_with_dets == 0:
        cap2 = cv2.VideoCapture(str(video))
        if cap2.isOpened():
            fd2 = init_mp_fd(model_selection=MP_MODEL_SELECTION, min_conf=MP_MIN_CONF)
            step = max(int(round(fps / SAMPLE_FPS)), 1)
            start_f = 0
            end_f = min(total_frames, int(WIN_MIN * fps))
            cap2.set(cv2.CAP_PROP_POS_FRAMES, start_f)
            ok, frame0nc = cap2.read()
            if ok and content0 is None:
                content0 = frame0nc.copy()
                h_c, w_c = content0.shape[:2]
                y_band = (0, h_c)

            for f in range(start_f, end_f, step):
                cap2.set(cv2.CAP_PROP_POS_FRAMES, f)
                ok, frame = cap2.read()
                if not ok:
                    break
                bboxes = detect_faces_mediapipe(fd2, frame)
                if not bboxes and USE_HAAR_FALLBACK:
                    bboxes = detect_faces_fallback_haar(frame)
                if bboxes:
                    frames_with_dets += 1
                    for (x1, y1, x2, y2) in bboxes:
                        xc = 0.5 * (x1 + x2)
                        tid = int(round(xc))
                        track_x.setdefault(tid, []).append(xc)
                        track_h.setdefault(tid, []).append(y2 - y1)
            if fd2 is not None:
                try:
                    fd2.close()
                except Exception:
                    pass
            cap2.release()

    cap.release()

    approx_total_samples = max(1, int(round(window_duration * SAMPLE_FPS)))
    min_frames = max(6, int(approx_total_samples * MIN_TRACK_FRAC))

    ids_len = [tid for tid, xs in track_x.items() if len(xs) >= min_frames]
    h_med = {tid: float(np.median(track_h.get(tid, [0]))) for tid in ids_len}
    if h_med:
        h_vals = np.array([h_med[tid] for tid in ids_len], float)
        h_thresh = float(np.percentile(h_vals, SIZE_PCTL))
        keep_ids = [tid for tid in ids_len if h_med.get(tid, 0.0) >= h_thresh]
    else:
        keep_ids = ids_len

    filtered_track_x = {tid: track_x[tid] for tid in keep_ids}

    stats_by_tid = {}
    for tid in keep_ids:
        xs = track_x.get(tid, [])
        ys = track_y.get(tid, [])
        hs = track_h.get(tid, [])
        stats_by_tid[tid] = {
            "len": int(len(xs)),
            "x_med_px": float(np.median(xs)) if xs else None,
            "y_med_px": float(np.median(ys)) if ys else None,
            "h_med_px": float(np.median(hs)) if hs else None,
        }

    if w_c is None:
        return False, {"session": session_dir.name, "error": "no frame size available"}
    seat_order, A, seam = compute_seat_order_and_A(
        filtered_track_x, frame_width=w_c, kappa=KAPPA, merge_deg=MERGE_DEG
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Gaze summary per seat-person (optional)
    # ------------------------------------------------------------
    persons = {}

    # Optional: reduziere Größe der JSON (sonst evtl. sehr groß)
    GAZE_EXPORT_STRIDE = 1  # 1 = alles exportieren, 2 = jedes 2. Sample, 5 = jedes 5. Sample
    GAZE_EXPORT_ROUND = 3  # Dezimalstellen (0 = int)

    for s in seat_order:
        yaws = track_yaw.get(s.track_id, [])

        # Downsample + runden (damit JSON klein bleibt)
        if yaws:
            yaws_ds = yaws[::GAZE_EXPORT_STRIDE]
            if GAZE_EXPORT_ROUND is not None and GAZE_EXPORT_ROUND >= 0:
                yaws_ds = [round(float(v), int(GAZE_EXPORT_ROUND)) for v in yaws_ds]
            else:
                yaws_ds = [float(v) for v in yaws_ds]
        else:
            yaws_ds = []

        if yaws_ds:
            y = np.asarray(yaws_ds, float)
            med = float(np.median(y))
            q1 = float(np.percentile(y, 25))
            q3 = float(np.percentile(y, 75))
            persons[str(int(s.person_id))] = {
                # Full time series (this is what you want)
                "yaw_deg_samples": yaws_ds,

                # Keep summary stats too (useful for debugging / fallback)
                "yaw_deg_median": med,
                "yaw_deg_iqr": float(q3 - q1),
                "n_samples": int(len(yaws_ds)),
            }
        else:
            persons[str(int(s.person_id))] = {
                "yaw_deg_samples": [],
                "yaw_deg_median": None,
                "yaw_deg_iqr": None,
                "n_samples": 0,
            }

    gazeout_json = {
        "session": session_dir.name,
        "method": "mediapipe_face_mesh_pnp" if (_MP_OK and fm is not None) else "none",
        "yaw_sign_convention": float(YAW_SIGN_CONVENTION),
        "export": {
            "stride": int(GAZE_EXPORT_STRIDE),
            "round_decimals": int(GAZE_EXPORT_ROUND),
        },
        "persons": persons,
    }

    (out_dir / "gaze_tracks.json").write_text(json.dumps(gazeout_json, indent=2), encoding="utf-8")


    seat_json = [{
        "person_id": s.person_id,
        "track_id": s.track_id,
        "theta_deg": round(s.theta_deg, 3),
        "neighbors": list(s.neighbors)
    } for s in seat_order]
    (out_dir / "seat_order.json").write_text(json.dumps(seat_json, indent=2), encoding="utf-8")

    adj = []
    for s in seat_order:
        i = s.person_id
        for j in s.neighbors:
            if i < j:
                adj.append({"u": i, "v": j})
    (out_dir / "adjacency.json").write_text(json.dumps(adj, indent=2), encoding="utf-8")

    np.save(out_dir / "A.npy", A)

    meta = {
        "video": str(video),
        "sample": {
            "win_starts": [float(x) for x in win_starts],
            "window_duration_sec": float(window_duration),
            "fps_nominal": float(fps),
            "frames_total": int(total_frames)
        },
        "content_crop": {"y1": int(y_band[0]), "y2": int(y_band[1]), "width": int(w_c), "height": int(h_c or 0)},
        "seam_deg": float(seam),
        "kappa": float(KAPPA),
        "merge_deg": float(MERGE_DEG),
        "tracks_raw": {str(int(k)): int(len(v)) for k, v in track_x.items()},
        "N_tracks_raw": int(len(track_x)),
        "N_tracks_len_ok": int(len(ids_len)),
        "N_tracks_size_ok": int(len(keep_ids)),
        "N_persons": int(len(seat_order)),
        "frames_with_dets": int(frames_with_dets),
        "min_frames_threshold": int(min_frames),
        "height_percentile_thresh": float(SIZE_PCTL),
        "mp_min_conf": float(MP_MIN_CONF),
        "mp_model_selection": int(MP_MODEL_SELECTION),
        "use_haar_fallback": bool(USE_HAAR_FALLBACK),
        "tracks_stats": {str(int(k)): v for k, v in stats_by_tid.items()},
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if content0 is None:
        cap0 = cv2.VideoCapture(str(video))
        ok0, f0 = cap0.read()
        cap0.release()
        if ok0:
            content0, _ = crop_letterbox(f0)
    if content0 is None:
        content0 = np.zeros((int(h_c or 512), int(w_c or 1024), 3), np.uint8)

    base = content0.copy()
    H, W = base.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    x_seam = int(((seam % 360.0) / 360.0) * W)
    cv2.line(base, (x_seam, 0), (x_seam, H - 1), (255, 0, 255), 2)
    cv2.putText(base, "seam", (min(x_seam + 6, W - 60), 24),
                font, 0.6, (255, 0, 255), 2, cv2.LINE_AA)

    def draw_wrap_line(img, p1, p2, color, thickness=2):
        (x1, y1), (x2, y2) = p1, p2
        if abs(x1 - x2) <= W / 2:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        else:
            if x2 > x1:
                x2s = x2 - W
            else:
                x2s = x2 + W
            cv2.line(img, (x1, y1), (x2s, y2), color, thickness)

    centers_px = {}
    for s in seat_order:
        tid = s.track_id
        st = stats_by_tid.get(tid, {})
        x_theta = int((((s.theta_deg + seam) % 360.0) / 360.0) * W)
        y_c = int(st.get("y_med_px", H * 0.45))
        h_med = float(st.get("h_med_px", H * 0.12))

        box_h = int(max(12, min(h_med, H * 0.35)))
        box_w = int(box_h * 0.8)
        x1 = max(0, x_theta - box_w // 2)
        x2 = min(W - 1, x_theta + box_w // 2)
        y1 = max(0, y_c - box_h // 2)
        y2 = min(H - 1, y_c + box_h // 2)

        cv2.rectangle(base, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = str(s.person_id)
        tsize, _ = cv2.getTextSize(label, font, 0.9, 2)
        cv2.rectangle(base, (x1, y1 - tsize[1] - 6),
                      (x1 + tsize[0] + 6, y1), (0, 255, 0), -1)
        cv2.putText(base, label, (x1 + 3, y1 - 6),
                    font, 0.9, (0, 0, 0), 2, cv2.LINE_AA)

        centers_px[s.person_id] = (x_theta, y_c)

    edges = set()
    for s in seat_order:
        i = s.person_id
        for j in s.neighbors:
            if i < j:
                edges.add((i, j))
    for (i, j) in edges:
        if i in centers_px and j in centers_px:
            draw_wrap_line(base, centers_px[i], centers_px[j], (0, 140, 255), 2)

    cv2.putText(base, "ID = Sitz-Nummer (clockwise). Orange lines = neighbours.",
                (10, H - 14), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_dir / "debug_labeled.jpg"), base)

    # Seat geometry + plots + dendrogram
    geom = ensure_seat_geometry_and_plots(
        out_dir=out_dir,
        seat_order=seat_order,
        seam_deg=seam,
        frame_width=w_c,
        recompute=RECOMPUTE_GEOMETRY,
    )

    summary = {
        "session": session_dir.name,
        "output_dir": str(out_dir),
        "N_persons": len(seat_order),
        "N_edges": len(adj),
        "N_tracks_raw": len(track_x),
        "N_tracks_len_ok": len(ids_len),
        "N_tracks_size_ok": len(keep_ids),
        "frames_with_dets": int(frames_with_dets),
        "seam_deg": float(seam),
        "N_persons_geom": int(len(geom["person_ids"])),
    }

    # ASD matching + groundtruth plot
    if ASD_ROOT is not None:
        asd_session_dir = ASD_ROOT / session_dir.name
        asd_info = match_asd_tracks_and_plot(
            geom=geom,
            asd_session_dir=asd_session_dir,
            frame_width=w_c,
            out_dir=out_dir,
        )
        summary.update({
            "N_asd_tracks": asd_info.get("N_asd_tracks", 0),
            "N_asd_used": asd_info.get("N_asd_used", 0),
            "N_asd_pairs": asd_info.get("N_pairs", 0),
        })

        spk_to_cluster = load_speaker_clusters(session_dir.name)
        if spk_to_cluster and asd_info.get("assignments"):
            _plot_groundtruth_clusters(
                path=out_dir / "seat_groundtruth_clusters.png",
                geom=geom,
                assignments=asd_info["assignments"],
                spk_to_cluster=spk_to_cluster,
            )

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return True, summary

# ----------------------------
# Session discovery & main
# ----------------------------
def find_sessions(vroot: Path) -> list[Path]:
    sessions = []
    for p in sorted(vroot.glob("session_*")):
        if (p / VIDEO_NAME).exists():
            sessions.append(p)
    return sessions

def append_index_line(index_path: Path, record: dict):
    with index_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def main(
    videos_root: Path | None = None,
    output_root: Path | None = None,
    asd_root: Path | None = None,
    labels_root: Path | None = None,
    recompute_geometry: bool = False,
):
    """
    Run seating prior pipeline.

    If arguments are None, use the module-level defaults:
      VIDEOS_ROOT, OUTPUT_ROOT, ASD_ROOT, LABELS_ROOT, RECOMPUTE_GEOMETRY.
    """
    global OUTPUT_ROOT, ASD_ROOT, LABELS_ROOT, RECOMPUTE_GEOMETRY

    # Fallbacks auf die oben gesetzten Konstanten
    videos_root = Path(videos_root) if videos_root is not None else VIDEOS_ROOT
    output_root = Path(output_root) if output_root is not None else OUTPUT_ROOT
    asd_root    = Path(asd_root)    if asd_root is not None    else ASD_ROOT
    labels_root = Path(labels_root) if labels_root is not None else LABELS_ROOT

    OUTPUT_ROOT = output_root
    ASD_ROOT = asd_root
    LABELS_ROOT = labels_root
    RECOMPUTE_GEOMETRY = bool(recompute_geometry)

    index_path = OUTPUT_ROOT / "sessions_index.jsonl"
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[seat_prior] videos_root = {videos_root}", flush=True)
    print(f"[seat_prior] output_root = {OUTPUT_ROOT}", flush=True)
    print(f"[seat_prior] asd_root   = {ASD_ROOT}", flush=True)
    print(f"[seat_prior] labels_root= {LABELS_ROOT}", flush=True)

    if not videos_root.exists():
        print(f"No such videos_root: {videos_root}", flush=True)
        return

    sessions = find_sessions(videos_root)
    if not sessions:
        print(f"No sessions with {VIDEO_NAME} found under {videos_root}", flush=True)
        return

    print(f"Found {len(sessions)} sessions", flush=True)
    for sdir in sessions:
        ok, res = process_video(sdir)
        if ok:
            print(
                f"[{res['session']}] persons={res['N_persons']} "
                f"(tracks_raw={res['N_tracks_raw']}, len_ok={res['N_tracks_len_ok']}, "
                f"size_ok={res['N_tracks_size_ok']}, frames_with_dets={res['frames_with_dets']}, "
                f"asd_pairs={res.get('N_asd_pairs', 0)}) → {res['output_dir']}",
                flush=True
            )
            append_index_line(index_path, res)
        else:
            err = res.get("error", "unknown error")
            print(f"[{sdir.name}] ERROR: {err}", flush=True)
            append_index_line(index_path, {"session": sdir.name, "error": err})



if __name__ == "__main__":
    main()
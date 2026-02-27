"""
distance_matrix_models.py

Typisierte Datenmodelle für Pipeline-Schritt 3 (Distanzmatrix je Session)

Ziel:
- distance_matrix: D (N x N) als numpy.ndarray
- distance_ids:    Liste von ID-Objekten (Reihenfolge ist verbindlich; 1:1 zu Zeilen/Spalten von D)
- optional:        distance_meta (Distanztyp, N, Fingerprints, Numerik-Checks, Timestamp)

Es enthält ausschließlich Modell-/Validierungs- und Datei-I/O-Logik
(keine CLI, keine Session-Iteration, keine hart kodierten Verzeichnisstrukturen).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


DEFAULT_DISTANCE_MATRIX_NAME = "distance_matrix.npy"
DEFAULT_DISTANCE_IDS_NAME = "distance_ids.json"
DEFAULT_DISTANCE_META_NAME = "distance_meta.json"


def _now_iso_utc() -> str:
    # ISO-Format ohne externe Abhängigkeiten (UTC, seconds precision)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_replace(src: Path, dst: Path) -> None:
    """
    Atomischer Replace (im selben Filesystem/Verzeichnis). Path.replace ist auf POSIX atomisch.
    """
    src.replace(dst)


def _atomic_write_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8")
        _atomic_replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _atomic_write_npy(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # WICHTIG: tmp MUSS auf ".npy" enden, sonst schreibt np.save nach "<tmp>.npy"
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.npy")
    try:
        np.save(tmp, arr)          # schreibt exakt tmp (weil tmp bereits .npy hat)
        _atomic_replace(tmp, path) # atomisch
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


@dataclass(frozen=True)
class DistanceMeta:
    """
    Optionale Metadaten für Schritt 3.

    Hinweise
    --------
    - input_fingerprint sollte idealerweise aus Schritt 2 (embed_meta.json) übernommen werden.
    - embeddings_normed kann genutzt werden, um zu dokumentieren, ob ein "schneller Pfad"
      (bereits normierte Embeddings) verwendet wurde.
    """
    distance_type: str = "cosine"
    n: int = 0
    created_at: str = ""
    input_fingerprint: Optional[str] = None
    model_name: Optional[str] = None
    embeddings_normed: Optional[bool] = None
    numeric: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "distance_type": self.distance_type,
            "n": int(self.n),
            "created_at": self.created_at or _now_iso_utc(),
        }
        if self.input_fingerprint is not None:
            d["input_fingerprint"] = self.input_fingerprint
        if self.model_name is not None:
            d["model_name"] = self.model_name
        if self.embeddings_normed is not None:
            d["embeddings_normed"] = bool(self.embeddings_normed)
        if self.numeric is not None:
            d["numeric"] = self.numeric
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "DistanceMeta":
        return DistanceMeta(
            distance_type=str(d.get("distance_type", "cosine")),
            n=int(d.get("n", 0)),
            created_at=str(d.get("created_at", "")),
            input_fingerprint=d.get("input_fingerprint"),
            model_name=d.get("model_name"),
            embeddings_normed=d.get("embeddings_normed"),
            numeric=d.get("numeric"),
        )


@dataclass
class DistanceMatrixSessionArtifacts:
    """
    Typisierte Repräsentation der Artefakte für Schritt 3 (pro Session).

    Attribute
    ---------
    distances:
        Distanzmatrix D mit Shape (N, N). Erwartet wird standardmäßig Cosine-Distanz.
    ids:
        Liste von ID-Objekten in exakt der Reihenfolge der Matrixzeilen/-spalten.
        Erwartetes Minimalformat je Eintrag: {"uid": "..."}.
        Optional: {"speaker_id": "..."} (kompatibel zu embedding_ids.json).
    meta:
        Optionales Metaobjekt.
    """
    distances: np.ndarray
    ids: list[dict[str, Any]]
    meta: Optional[DistanceMeta] = None

    def validate(self, *, strict_numeric_checks: bool = True) -> None:
        if not isinstance(self.distances, np.ndarray):
            raise TypeError("distances muss ein numpy.ndarray sein.")
        if self.distances.ndim != 2:
            raise ValueError(
                f"D muss 2D sein (N x N). Gefunden: ndim={self.distances.ndim}, shape={self.distances.shape}"
            )
        n0, n1 = self.distances.shape
        if n0 != n1:
            raise ValueError(f"D muss quadratisch sein (N x N). Gefunden: shape={self.distances.shape}")

        if not isinstance(self.ids, list):
            raise TypeError("ids muss eine Liste von Dicts sein (z. B. aus embedding_ids.json).")
        if len(self.ids) != n0:
            raise ValueError(f"Inkonsistenz: len(ids) ({len(self.ids)}) != D.shape[0] ({n0}).")

        # Minimalvalidierung IDs
        for i, item in enumerate(self.ids):
            if not isinstance(item, dict):
                raise TypeError(f"ids[{i}] muss ein dict sein, gefunden: {type(item)}")
            if "uid" not in item:
                raise ValueError(f"ids[{i}] fehlt Pflichtfeld 'uid'.")

        if strict_numeric_checks and n0 > 0:
            D = self.distances
            if not np.isfinite(D).all():
                raise ValueError("D enthält NaN/Inf.")

            # Diagonale ~ 0
            diag = np.diag(D)
            if not np.allclose(diag, 0.0, atol=1e-6):
                raise ValueError("D hat keine (nahezu) 0-Diagonale (tol=1e-6).")

            # Symmetrie ~
            if not np.allclose(D, D.T, atol=1e-6):
                raise ValueError("D ist nicht (nahezu) symmetrisch (tol=1e-6).")

            # Plausibilitätscheck (Cosine-Distanz typischerweise in [0, 2])
            # Nicht hart erzwingen, aber auffällige Werte als Fehler behandeln.
            if D.min() < -1e-3 or D.max() > 2.0 + 1e-3:
                raise ValueError(f"D Wertebereich unplausibel: min={float(D.min()):.6f}, max={float(D.max()):.6f}")

    def to_files(
        self,
        out_dir: str | Path,
        *,
        distance_matrix_name: str = DEFAULT_DISTANCE_MATRIX_NAME,
        distance_ids_name: str = DEFAULT_DISTANCE_IDS_NAME,
        distance_meta_name: str = DEFAULT_DISTANCE_META_NAME,
        overwrite: bool = True,
        write_meta: bool = True,
        strict_numeric_checks: bool = True,
    ) -> dict[str, Path]:
        """
        Speichert Distanz-Artefakte in ein Ausgabeverzeichnis (atomisch).

        Rückgabe:
            dict mit Keys: "distance_matrix", "distance_ids" (+ optional "distance_meta")
        """
        self.validate(strict_numeric_checks=strict_numeric_checks)

        out_path = Path(out_dir).expanduser().resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        p_matrix = out_path / distance_matrix_name
        p_ids = out_path / distance_ids_name
        p_meta = out_path / distance_meta_name

        if not overwrite:
            for p in (p_matrix, p_ids, p_meta):
                if p.exists():
                    raise FileExistsError(f"Datei existiert bereits: {p}")

        _atomic_write_npy(p_matrix, self.distances)
        _atomic_write_json(p_ids, self.ids)

        written: dict[str, Path] = {
            "distance_matrix": p_matrix,
            "distance_ids": p_ids,
        }

        if write_meta:
            meta_obj = self.meta or DistanceMeta(
                distance_type="cosine",
                n=int(self.distances.shape[0]),
                created_at=_now_iso_utc(),
            )
            _atomic_write_json(p_meta, meta_obj.to_dict())
            written["distance_meta"] = p_meta

        return written

    @staticmethod
    def from_files(
        in_dir: str | Path,
        *,
        distance_matrix_name: str = DEFAULT_DISTANCE_MATRIX_NAME,
        distance_ids_name: str = DEFAULT_DISTANCE_IDS_NAME,
        distance_meta_name: str = DEFAULT_DISTANCE_META_NAME,
        mmap_mode: Optional[str] = None,
        load_meta: bool = True,
        strict_numeric_checks: bool = True,
    ) -> "DistanceMatrixSessionArtifacts":
        """
        Lädt Distanz-Artefakte aus einem Verzeichnis.
        """
        in_path = Path(in_dir).expanduser().resolve()
        p_matrix = in_path / distance_matrix_name
        p_ids = in_path / distance_ids_name
        p_meta = in_path / distance_meta_name

        if not p_matrix.is_file():
            raise FileNotFoundError(f"Distanzmatrix-Datei nicht gefunden: {p_matrix}")
        if not p_ids.is_file():
            raise FileNotFoundError(f"IDs-Datei nicht gefunden: {p_ids}")

        D = np.load(p_matrix, mmap_mode=mmap_mode)
        ids = json.loads(p_ids.read_text(encoding="utf-8"))

        meta: Optional[DistanceMeta] = None
        if load_meta and p_meta.is_file():
            meta_dict = json.loads(p_meta.read_text(encoding="utf-8"))
            if isinstance(meta_dict, dict):
                meta = DistanceMeta.from_dict(meta_dict)

        obj = DistanceMatrixSessionArtifacts(distances=D, ids=ids, meta=meta)
        obj.validate(strict_numeric_checks=strict_numeric_checks)
        return obj

"""
embed_texts_models.py

Typisierte Datenmodelle für Pipeline-Schritt 2 (Embeddings je Session).

Ziel:
- pro Session:
  - embeddings.npy        : Numpy-Matrix (Shape: n_speakers x d)
  - embedding_ids.json    : Zeilen-Mapping (Reihenfolge muss exakt zu embeddings.npy passen)
  - embed_meta.json       : optionale Metadaten (Modell, Dim, Counts, Timestamp, etc.)
- zusätzlich (run-weit, in Service/Orchestrator geschrieben):
  - run_meta_embed_texts.json

Abgrenzung:
- Keine CLI-Logik.
- Keine Dateisystem-Suche (nur explizite Datei-/Verzeichnisparameter).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import json
import os
import tempfile

import numpy as np

PathLike = Union[str, Path]


def _atomic_write_json(path: Path, data: Any) -> None:
    """
    Atomisches Schreiben von JSON:
    - schreibt in temporäre Datei im gleichen Verzeichnis
    - ersetzt danach per os.replace
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    finally:
        # Falls os.replace fehlschlägt, tmp aufräumen
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


def _ensure_2d(a: np.ndarray, name: str) -> None:
    if not isinstance(a, np.ndarray):
        raise TypeError(f"{name} muss ein numpy.ndarray sein.")
    if a.ndim != 2:
        raise ValueError(f"{name} muss 2D sein (n x d), ist aber {a.ndim}D.")


@dataclass(frozen=True)
class EmbeddingRowId:
    """
    Mapping eines Embedding-Zeilenindex auf den Sprecher.

    Mindestanforderung:
    - uid: eindeutiger Identifier (z. B. 'session_132_spk_0')

    Optional:
    - speaker_id: sprecherspezifische ID (z. B. 'spk_0')
    """

    uid: str
    speaker_id: Optional[str] = None

    def validate(self) -> None:
        if not isinstance(self.uid, str) or not self.uid.strip():
            raise ValueError("EmbeddingRowId.uid muss ein nicht-leerer String sein.")
        if self.speaker_id is not None and (not isinstance(self.speaker_id, str) or not self.speaker_id.strip()):
            raise ValueError("EmbeddingRowId.speaker_id muss None oder ein nicht-leerer String sein.")


@dataclass(frozen=True)
class EmbedMeta:
    """
    Optionale Metadaten für die Embedding-Erzeugung (pro Session).
    """

    model_name: str
    embedding_dim: int
    created_at: str  # ISO 8601 string
    n_rows: int
    device: Optional[str] = None
    batch_size: Optional[int] = None
    input_fingerprint: Optional[str] = None  # z. B. Hash über Input-Texte/IDs
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def validate(self) -> None:
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("EmbedMeta.model_name muss ein nicht-leerer String sein.")
        if not isinstance(self.embedding_dim, int) or self.embedding_dim < 0:
            raise ValueError("EmbedMeta.embedding_dim muss eine nicht-negative Ganzzahl sein.")
        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ValueError("EmbedMeta.created_at muss ein nicht-leerer String sein (ISO 8601).")
        if not isinstance(self.n_rows, int) or self.n_rows < 0:
            raise ValueError("EmbedMeta.n_rows muss eine nicht-negative Ganzzahl sein.")
        if self.device is not None and (not isinstance(self.device, str) or not self.device.strip()):
            raise ValueError("EmbedMeta.device muss None oder ein nicht-leerer String sein.")
        if self.batch_size is not None and (not isinstance(self.batch_size, int) or self.batch_size <= 0):
            raise ValueError("EmbedMeta.batch_size muss None oder eine positive Ganzzahl sein.")
        if self.input_fingerprint is not None and (
            not isinstance(self.input_fingerprint, str) or not self.input_fingerprint.strip()
        ):
            raise ValueError("EmbedMeta.input_fingerprint muss None oder ein nicht-leerer String sein.")
        if not isinstance(self.warnings, list) or not all(isinstance(x, str) for x in self.warnings):
            raise TypeError("EmbedMeta.warnings muss list[str] sein.")
        if not isinstance(self.errors, list) or not all(isinstance(x, str) for x in self.errors):
            raise TypeError("EmbedMeta.errors muss list[str] sein.")


@dataclass(frozen=True)
class EmbedSessionArtifacts:
    """
    Ergebnisartefakte für eine Session.

    - row_ids: Mapping der Zeilenreihenfolge zu Sprecher-IDs
    - embeddings: Matrix (n x d)
    - meta: optionale Metadaten

    Standard-Dateinamen (pro Session-Verzeichnis):
    - embeddings.npy
    - embedding_ids.json
    - embed_meta.json (optional)
    """

    row_ids: List[EmbeddingRowId]
    embeddings: np.ndarray
    meta: Optional[EmbedMeta] = None

    def validate(self) -> None:
        if not isinstance(self.row_ids, list):
            raise TypeError("row_ids muss eine Liste sein.")
        for r in self.row_ids:
            if not isinstance(r, EmbeddingRowId):
                raise TypeError("row_ids muss Elemente vom Typ EmbeddingRowId enthalten.")
            r.validate()

        _ensure_2d(self.embeddings, "embeddings")

        n = int(self.embeddings.shape[0])
        if len(self.row_ids) != n:
            raise ValueError(f"Mismatch: len(row_ids)={len(self.row_ids)} passt nicht zu embeddings.shape[0]={n}.")

        # UID-Eindeutigkeit (stabiler Schlüssel)
        uids = [r.uid for r in self.row_ids]
        if len(set(uids)) != len(uids):
            raise ValueError("row_ids enthält doppelte uid-Werte (UIDs müssen je Session eindeutig sein).")

        # Optional: Metadatenkonsistenz
        if self.meta is not None:
            if not isinstance(self.meta, EmbedMeta):
                raise TypeError("meta muss None oder EmbedMeta sein.")
            self.meta.validate()
            if self.meta.n_rows != n:
                raise ValueError(f"meta.n_rows={self.meta.n_rows} passt nicht zu embeddings.shape[0]={n}.")
            d = int(self.embeddings.shape[1])
            if self.meta.embedding_dim != d:
                raise ValueError(f"meta.embedding_dim={self.meta.embedding_dim} passt nicht zu embeddings.shape[1]={d}.")

    def to_session_dir(
        self,
        session_dir: PathLike,
        *,
        embeddings_filename: str = "embeddings.npy",
        ids_filename: str = "embedding_ids.json",
        meta_filename: str = "embed_meta.json",
        write_meta: bool = True,
    ) -> None:
        """
        Schreibt Artefakte in ein Session-Verzeichnis.
        """
        self.validate()

        out_path = Path(session_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        emb_path = out_path / embeddings_filename
        ids_path = out_path / ids_filename
        meta_path = out_path / meta_filename

        # embeddings.npy
        # deterministisch: keine Pickles
        np.save(emb_path, self.embeddings, allow_pickle=False)

        # embedding_ids.json (Liste von Dicts)
        ids_payload = [asdict(r) for r in self.row_ids]
        _atomic_write_json(ids_path, ids_payload)

        # embed_meta.json (optional)
        if write_meta and self.meta is not None:
            _atomic_write_json(meta_path, asdict(self.meta))

    @classmethod
    def from_session_dir(
        cls,
        session_dir: PathLike,
        *,
        embeddings_filename: str = "embeddings.npy",
        ids_filename: str = "embedding_ids.json",
        meta_filename: str = "embed_meta.json",
        load_meta: bool = True,
    ) -> "EmbedSessionArtifacts":
        """
        Lädt Artefakte aus einem Session-Verzeichnis.
        """
        base = Path(session_dir)
        emb_path = base / embeddings_filename
        ids_path = base / ids_filename
        meta_path = base / meta_filename

        if not ids_path.is_file():
            raise FileNotFoundError(f"ID-Datei nicht gefunden: {ids_path}")
        if not emb_path.is_file():
            raise FileNotFoundError(f"Embeddings-Datei nicht gefunden: {emb_path}")

        with ids_path.open("r", encoding="utf-8") as f:
            raw_ids = json.load(f)

        if not isinstance(raw_ids, list):
            raise ValueError("embedding_ids.json muss eine JSON-Liste enthalten.")

        row_ids: List[EmbeddingRowId] = []
        for i, item in enumerate(raw_ids):
            if not isinstance(item, dict):
                raise ValueError(f"embedding_ids.json: Element {i} ist kein Objekt.")
            uid = item.get("uid")
            speaker_id = item.get("speaker_id", None)
            row_ids.append(EmbeddingRowId(uid=uid, speaker_id=speaker_id))

        embeddings = np.load(emb_path, allow_pickle=False)

        meta: Optional[EmbedMeta] = None
        if load_meta and meta_path.is_file():
            with meta_path.open("r", encoding="utf-8") as f:
                raw_meta = json.load(f)
            if not isinstance(raw_meta, dict):
                raise ValueError("embed_meta.json muss ein JSON-Objekt enthalten.")
            meta = EmbedMeta(
                model_name=raw_meta.get("model_name"),
                embedding_dim=int(raw_meta.get("embedding_dim")),
                created_at=raw_meta.get("created_at"),
                n_rows=int(raw_meta.get("n_rows")),
                device=raw_meta.get("device"),
                batch_size=raw_meta.get("batch_size"),
                input_fingerprint=raw_meta.get("input_fingerprint"),
                warnings=list(raw_meta.get("warnings", [])),
                errors=list(raw_meta.get("errors", [])),
            )

        obj = cls(row_ids=row_ids, embeddings=embeddings, meta=meta)
        obj.validate()
        return obj


@dataclass(frozen=True)
class EmbedTextsRunMeta:
    """
    Run-weite Metadaten für Schritt 2 (über alle Sessions).
    Wird typischerweise als out_dir/run_meta_embed_texts.json geschrieben.
    """

    schema_version: int
    created_at: str  # ISO 8601 string
    model_name: str
    device: Optional[str]
    batch_size: int
    force: bool

    sessions_total: int
    sessions_embedded: int
    sessions_skipped: int

    speakers_total: int
    speakers_embedded: int
    speakers_skipped: int

    skipped_sessions: List[Dict[str, Any]] = field(default_factory=list)  # {session_name, reason}
    errors: List[Dict[str, Any]] = field(default_factory=list)  # {session_name, error}

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError("EmbedTextsRunMeta.schema_version muss 1 sein.")
        if not isinstance(self.created_at, str) or not self.created_at.strip():
            raise ValueError("created_at muss ein nicht-leerer String sein (ISO 8601).")
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name muss ein nicht-leerer String sein.")
        if self.device is not None and (not isinstance(self.device, str) or not self.device.strip()):
            raise ValueError("device muss None oder ein nicht-leerer String sein.")
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise ValueError("batch_size muss eine positive Ganzzahl sein.")
        if not isinstance(self.force, bool):
            raise TypeError("force muss bool sein.")

        for k in (
            "sessions_total",
            "sessions_embedded",
            "sessions_skipped",
            "speakers_total",
            "speakers_embedded",
            "speakers_skipped",
        ):
            v = getattr(self, k)
            if not isinstance(v, int) or v < 0:
                raise ValueError(f"{k} muss eine nicht-negative Ganzzahl sein.")

        if not isinstance(self.skipped_sessions, list) or not all(isinstance(x, dict) for x in self.skipped_sessions):
            raise TypeError("skipped_sessions muss list[dict] sein.")
        if not isinstance(self.errors, list) or not all(isinstance(x, dict) for x in self.errors):
            raise TypeError("errors muss list[dict] sein.")

    def to_file(self, path: PathLike) -> None:
        self.validate()
        _atomic_write_json(Path(path), asdict(self))

    @classmethod
    def from_file(cls, path: PathLike) -> "EmbedTextsRunMeta":
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"Run-Meta-Datei nicht gefunden: {p}")
        with p.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError("run_meta_embed_texts.json muss ein JSON-Objekt enthalten.")
        obj = cls(
            schema_version=int(raw.get("schema_version")),
            created_at=raw.get("created_at"),
            model_name=raw.get("model_name"),
            device=raw.get("device"),
            batch_size=int(raw.get("batch_size")),
            force=bool(raw.get("force")),
            sessions_total=int(raw.get("sessions_total")),
            sessions_embedded=int(raw.get("sessions_embedded")),
            sessions_skipped=int(raw.get("sessions_skipped")),
            speakers_total=int(raw.get("speakers_total")),
            speakers_embedded=int(raw.get("speakers_embedded")),
            speakers_skipped=int(raw.get("speakers_skipped")),
            skipped_sessions=list(raw.get("skipped_sessions", [])),
            errors=list(raw.get("errors", [])),
        )
        obj.validate()
        return obj

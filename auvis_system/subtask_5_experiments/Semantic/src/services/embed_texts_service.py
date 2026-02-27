"""
embed_texts_service.py

Service für Pipeline-Schritt 2:
- Liest pro Session: out_dir/session_*/extracted_texts.json (Schema v1 aus Schritt 1)
- Erzeugt Speaker-Level-Embeddings (SentenceTransformers), deterministisch sortiert
- Schreibt pro Session:
  - embeddings.npy
  - embedding_ids.json
  - optional embed_meta.json
- Schreibt zusätzlich (run-weit):
  - out_dir/run_meta_embed_texts.json

Abgrenzung:
- Keine CLI-Logik (kein argparse).
- Keine Projektpfad-Suche (Session-Liste wird vom Orchestrator übergeben).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import hashlib
import json
import logging

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "Das Paket 'sentence-transformers' ist nicht installiert. "
        "Bitte installiere es in deiner venv (z. B. pip install sentence-transformers)."
    ) from e

from semantic.models.embed_texts_models import (
    EmbedMeta,
    EmbedSessionArtifacts,
    EmbedTextsRunMeta,
    EmbeddingRowId,
)

PathLike = Union[str, Path]

logger = logging.getLogger(__name__)


# -----------------------------
# Helpers
# -----------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _pick_device(device: Optional[str]) -> str:
    """
    Default-Device-Entscheidung:
    - Wenn device gesetzt: übernehmen
    - Sonst: cuda, falls torch verfügbar und cuda nutzbar; andernfalls cpu
    """
    if device is not None:
        return device

    try:
        import torch  # noqa: WPS433
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _stable_speaker_sort_key(speaker_id: Optional[str], uid: str) -> Tuple[int, str]:
    """
    Deterministische Sortierung:
    - bevorzugt numerisches spk_<n> nach n
    - sonst lexikographisch nach speaker_id/uid
    """
    if isinstance(speaker_id, str) and speaker_id.startswith("spk_"):
        tail = speaker_id[4:]
        if tail.isdigit():
            return (0, f"{int(tail):08d}")
    # fallback: stabil und deterministisch
    return (1, speaker_id or uid)


def _fingerprint_inputs(row_ids: Sequence[EmbeddingRowId], texts: Sequence[str], model_name: str) -> str:
    """
    Stabiler Fingerprint über (model_name, uid, speaker_id, text).
    Für Cache/Skip-Entscheidungen nutzbar.
    """
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    for r, t in zip(row_ids, texts):
        h.update(b"\0")
        h.update((r.uid or "").encode("utf-8"))
        h.update(b"\0")
        h.update((r.speaker_id or "").encode("utf-8"))
        h.update(b"\0")
        h.update(t.encode("utf-8"))
    return h.hexdigest()


def _load_extracted_texts_session(extracted_texts_path: PathLike) -> Tuple[str, List[EmbeddingRowId], List[str]]:
    """
    Lädt eine extracted_texts.json (Schema v1) und extrahiert Speaker-Level inputs:
    - session_name
    - row_ids: (uid, speaker_id)
    - texts: speaker.text
    """
    p = Path(extracted_texts_path)
    if not p.is_file():
        raise FileNotFoundError(f"extracted_texts.json nicht gefunden: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Ungültiges JSON-Format in {p}: erwartet Objekt.")

    schema_version = data.get("schema_version")
    if schema_version != 1:
        raise ValueError(f"Unsupported schema_version={schema_version} in {p} (erwartet 1).")

    session_name = data.get("session_name")
    if not isinstance(session_name, str) or not session_name.strip():
        raise ValueError(f"session_name fehlt/leer in {p}.")

    speakers = data.get("speakers", [])
    if not isinstance(speakers, list):
        raise ValueError(f"speakers muss Liste sein in {p}.")

    rows: List[Tuple[EmbeddingRowId, str]] = []
    seen_uid: set[str] = set()

    for idx, spk in enumerate(speakers):
        if not isinstance(spk, dict):
            continue
        uid = spk.get("uid")
        speaker_id = spk.get("speaker_id")
        text = spk.get("text")

        if not isinstance(uid, str) or not uid.strip():
            raise ValueError(f"{p}: speakers[{idx}].uid fehlt/leer.")
        if uid in seen_uid:
            raise ValueError(f"{p}: Duplicate uid: {uid}")
        seen_uid.add(uid)

        # speaker_id optional, aber falls vorhanden: nicht-leer
        if speaker_id is not None and (not isinstance(speaker_id, str) or not speaker_id.strip()):
            raise ValueError(f"{p}: speakers[{idx}].speaker_id ist gesetzt aber leer/ungültig.")

        text_str = "" if text is None else str(text).strip()
        if text_str == "":
            # Leere Texte werden übersprungen, aber später als warning gemeldet
            continue

        rid = EmbeddingRowId(uid=uid, speaker_id=speaker_id)
        rows.append((rid, text_str))

    # deterministisch sortieren
    rows.sort(key=lambda x: _stable_speaker_sort_key(x[0].speaker_id, x[0].uid))

    row_ids = [r for r, _ in rows]
    texts = [t for _, t in rows]
    return session_name, row_ids, texts


# -----------------------------
# Public API
# -----------------------------
@dataclass(frozen=True)
class EmbedConfig:
    model_name: str = "paraphrase-multilingual-mpnet-base-v2"
    batch_size: int = 32
    device: Optional[str] = None
    normalize_embeddings: bool = False
    force: bool = False
    write_meta: bool = True


def embed_sessions(
    session_dirs: Sequence[PathLike],
    *,
    out_dir: PathLike,
    config: EmbedConfig = EmbedConfig(),
) -> EmbedTextsRunMeta:
    """
    Orchestrierbare Service-Funktion für Schritt 2 (sessionweise).
    Erwartet eine Liste von Session-Verzeichnissen im out_dir (z. B. out_dir/session_132).

    Jeder session_dir muss eine extracted_texts.json enthalten.
    """
    if config.batch_size <= 0:
        raise ValueError("batch_size muss > 0 sein.")

    out_base = Path(out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    resolved_device = _pick_device(config.device)

    # Modell genau einmal laden (Performance)
    model = SentenceTransformer(config.model_name, device=resolved_device)

    sessions_total = 0
    sessions_embedded = 0
    sessions_skipped = 0

    speakers_total = 0
    speakers_embedded = 0
    speakers_skipped = 0

    skipped_sessions: List[Dict[str, str]] = []
    errors: List[Dict[str, str]] = []

    for sdir_like in session_dirs:
        sessions_total += 1
        sdir = Path(sdir_like)

        extracted_path = sdir / "extracted_texts.json"
        session_name = sdir.name

        try:
            session_name_from_file, row_ids, texts = _load_extracted_texts_session(extracted_path)
            # Konsistenz: falls Pfadname != JSON-Sessionname, nur warnen (nicht fatal)
            if session_name_from_file != session_name:
                logger.warning(
                    "Session-Name mismatch: dir=%s json=%s", session_name, session_name_from_file
                )
                session_name = session_name_from_file

            speakers_total += len(row_ids)

            # Skip: keine gültigen Texte
            if len(row_ids) == 0:
                sessions_skipped += 1
                skipped_sessions.append({"session_name": session_name, "reason": "no_nonempty_speaker_texts"})
                logger.info("Skip %s: keine nicht-leeren Speaker-Texte.", session_name)
                continue

            # Cache/Overwrite-Logik pro Session
            emb_path = sdir / "embeddings.npy"
            ids_path = sdir / "embedding_ids.json"
            meta_path = sdir / "embed_meta.json"

            input_fp = _fingerprint_inputs(row_ids, texts, config.model_name)

            if (not config.force) and emb_path.is_file() and ids_path.is_file():
                # optional: wenn meta existiert und fingerprint gleich, skip
                if meta_path.is_file():
                    try:
                        with meta_path.open("r", encoding="utf-8") as f:
                            raw_meta = json.load(f)
                        old_fp = raw_meta.get("input_fingerprint")
                        old_model = raw_meta.get("model_name")
                        if old_fp == input_fp and old_model == config.model_name:
                            sessions_skipped += 1
                            speakers_skipped += len(row_ids)
                            skipped_sessions.append({"session_name": session_name, "reason": "cached_unchanged"})
                            logger.info("Skip %s: cached_unchanged.", session_name)
                            continue
                    except Exception:
                        # falls meta kaputt: nicht skippen, neu schreiben
                        pass

            # Embedding berechnen
            embs: List[np.ndarray] = []
            for start in range(0, len(texts), config.batch_size):
                batch = texts[start : start + config.batch_size]
                batch_emb = model.encode(
                    batch,
                    batch_size=len(batch),
                    normalize_embeddings=config.normalize_embeddings,
                    show_progress_bar=False,
                )
                embs.append(np.asarray(batch_emb, dtype=np.float32))

            embeddings = np.vstack(embs)
            if embeddings.shape[0] != len(row_ids):
                raise ValueError(
                    f"Embedding count mismatch for {session_name}: "
                    f"rows={len(row_ids)} vs emb_rows={embeddings.shape[0]}"
                )

            meta = EmbedMeta(
                model_name=config.model_name,
                embedding_dim=int(embeddings.shape[1]),
                created_at=_utc_now_iso(),
                n_rows=int(embeddings.shape[0]),
                device=resolved_device,
                batch_size=config.batch_size,
                input_fingerprint=input_fp,
                warnings=[],
                errors=[],
            )

            artifacts = EmbedSessionArtifacts(row_ids=list(row_ids), embeddings=embeddings, meta=meta)
            artifacts.to_session_dir(sdir, write_meta=config.write_meta)

            sessions_embedded += 1
            speakers_embedded += len(row_ids)
            logger.info("Embedded %s: speakers=%d dim=%d", session_name, len(row_ids), embeddings.shape[1])

        except Exception as e:
            sessions_skipped += 1
            errors.append({"session_name": session_name, "error": f"{type(e).__name__}: {e}"})
            skipped_sessions.append({"session_name": session_name, "reason": "error"})
            logger.exception("Fehler in embed_sessions für %s", session_name)

    run_meta = EmbedTextsRunMeta(
        schema_version=1,
        created_at=_utc_now_iso(),
        model_name=config.model_name,
        device=resolved_device,
        batch_size=config.batch_size,
        force=bool(config.force),
        sessions_total=sessions_total,
        sessions_embedded=sessions_embedded,
        sessions_skipped=sessions_skipped,
        speakers_total=speakers_total,
        speakers_embedded=speakers_embedded,
        speakers_skipped=speakers_skipped,
        skipped_sessions=skipped_sessions,
        errors=errors,
    )

    # Persistenz des Run-Meta ist Aufgabe des Orchestrators; hier nur Rückgabe.
    run_meta.validate()
    return run_meta

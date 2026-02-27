# src/semantic/services/extract_texts_service.py
"""
Service: Textextraktion aus Baseline-*System*-Hypothesen (.vtt) – konsequent sessionweise.

Ziel (Schritt 1):
- Liest NICHT aus `labels/` (Ground Truth), sondern aus `output/` (Baseline-Hypothesen).
- baseline_dir kann entweder
  (a) ein Split-Ordner sein (enthält session_*), oder
  (b) ein einzelner Session-Ordner (z. B. .../session_40).

Output (empfohlen pro Session):
- out_dir/session_*/extracted_texts.json (wird durch `write_session_extracted_texts(...)` erzeugt)

Robustheit:
- Deterministische Sortierung (Sessions, Speaker-Dateien, Segmente)
- Sauberes Logging (fehlende Ordner, leere VTTs, Parse-Probleme)
- Fehlertolerantes VTT-Parsing (Encoding-Fallback, NOTE/STYLE/REGION ignorieren)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from src.semantic.models.extract_texts_models import (
    SCHEMA_VERSION,
    SessionExtractedTexts,
    SpeakerExtractedText,
    VttCueSegment,
)

LOGGER = logging.getLogger(__name__)

_TIME_RE = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(?P<end>\d{2}:\d{2}:\d{2}\.\d{3})(?:\s+.*)?$"
)


# -----------------------------
# Public API
# -----------------------------

def list_session_dirs(baseline_dir: str | Path) -> List[Tuple[str, Path]]:
    """
    Ermittelt Session-Verzeichnisse deterministisch.

    Rückgabe:
      Liste von (session_name, session_path), nach session_name sortiert.
    """
    base = Path(baseline_dir)
    if not base.exists():
        raise FileNotFoundError(f"baseline_dir existiert nicht: {base}")

    if base.is_dir() and base.name.startswith("session_"):
        return [(base.name, base.resolve())]

    session_dirs = sorted([p for p in base.glob("session_*") if p.is_dir()], key=lambda p: p.name)
    return [(p.name, p.resolve()) for p in session_dirs]


def extract_session_texts(
    session_dir: str | Path,
    *,
    output_dir_name: str = "output",
    include_segments: bool = True,
) -> SessionExtractedTexts:
    """
    Extrahiert Texte für genau eine Session.

    - Liest speaker-VTTs aus: session_dir/output_dir_name/spk_*.vtt
    - Erzeugt SpeakerExtractedText pro Speaker.
    - Session wird auch dann zurückgegeben, wenn keine Speaker-Texte extrahiert wurden (speakers=[]).
    """
    sess_path = Path(session_dir)
    if not sess_path.exists() or not sess_path.is_dir():
        raise FileNotFoundError(f"Session-Ordner existiert nicht: {sess_path}")

    session_name = sess_path.name
    out_path = sess_path / output_dir_name

    if not out_path.exists() or not out_path.is_dir():
        LOGGER.warning(
            "Session %s: Output-Ordner fehlt (%s). Session wird mit speakers=[] erzeugt.",
            session_name,
            out_path,
        )
        return SessionExtractedTexts(session_name=session_name, speakers=[], schema_version=SCHEMA_VERSION)

    vtt_files = _list_speaker_vtts(out_path)
    if not vtt_files:
        LOGGER.warning(
            "Session %s: Keine Speaker-VTTs gefunden unter %s (Pattern spk_*.vtt).",
            session_name,
            out_path,
        )
        return SessionExtractedTexts(session_name=session_name, speakers=[], schema_version=SCHEMA_VERSION)

    speakers: List[SpeakerExtractedText] = []
    for vtt_file in vtt_files:
        speaker_id = vtt_file.stem  # z.B. "spk_0"
        uid = f"{session_name}_{speaker_id}"

        try:
            segments = _parse_vtt_to_segments(vtt_file)
        except Exception as e:
            LOGGER.exception("Session %s / %s: Fehler beim Parsen (%s): %s", session_name, speaker_id, vtt_file, e)
            segments = []

        # Aggregierter Text (deterministisch aus cue_index-Reihenfolge)
        joined = " ".join([s.text for s in segments if s.text]).strip()

        if not joined and not segments:
            # Leere/unerwartete Datei: wir loggen, aber erzeugen keinen Speaker-Eintrag.
            LOGGER.info("Session %s / %s: Keine verwertbaren Inhalte in %s.", session_name, speaker_id, vtt_file)
            continue

        if not include_segments:
            segments = []

        speakers.append(
            SpeakerExtractedText(
                speaker_id=speaker_id,
                uid=uid,
                text=joined,
                segments=segments,
            )
        )

    if not speakers:
        LOGGER.warning(
            "Session %s: Keine Speaker-Texte extrahiert (alle VTTs leer/unlesbar).",
            session_name,
        )

    return SessionExtractedTexts(session_name=session_name, speakers=speakers, schema_version=SCHEMA_VERSION)


def extract_texts_by_session(
    baseline_dir: str | Path,
    *,
    output_dir_name: str = "output",
    include_segments: bool = True,
) -> List[SessionExtractedTexts]:
    """
    Extrahiert Texte für alle Sessions unter baseline_dir.

    Rückgabe:
      Liste von SessionExtractedTexts, deterministisch nach session_name sortiert.
    """
    sessions = list_session_dirs(baseline_dir)
    if not sessions:
        LOGGER.warning("Keine session_* Ordner gefunden unter %s.", baseline_dir)
        return []

    results: List[SessionExtractedTexts] = []
    for session_name, session_path in sessions:
        _ = session_name  # nur zur Lesbarkeit; Name steckt auch in session_path
        results.append(
            extract_session_texts(
                session_path,
                output_dir_name=output_dir_name,
                include_segments=include_segments,
            )
        )
    # deterministische Ordnung
    return sorted(results, key=lambda x: x.session_name)


def write_session_extracted_texts(
    extracted: SessionExtractedTexts,
    *,
    out_dir: str | Path,
    filename: str = "extracted_texts.json",
) -> Path:
    """
    Persistiert genau eine Session in die Zielstruktur:
      out_dir/<session_name>/<filename>
    """
    base = Path(out_dir)
    target_dir = base / extracted.session_name
    target_dir.mkdir(parents=True, exist_ok=True)

    target_file = target_dir / filename
    payload = extracted.to_dict()

    with target_file.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return target_file


# -----------------------------
# Internals
# -----------------------------

def _list_speaker_vtts(output_dir: Path) -> List[Path]:
    """
    Findet spk_*.vtt deterministisch.

    Sortierung:
      - bevorzugt numerisch nach spk_<n>
      - fallback lexikographisch
    """
    files = [p for p in output_dir.glob("spk_*.vtt") if p.is_file()]

    def _key(p: Path) -> tuple:
        m = re.match(r"^spk_(\d+)$", p.stem)
        if m:
            return (0, int(m.group(1)))
        return (1, p.stem)

    return sorted(files, key=_key)


def _read_text_file_best_effort(path: Path) -> str:
    """
    Robust gegen BOM/Encoding-Probleme.
    """
    try:
        return path.read_text(encoding="utf-8-sig", errors="strict")
    except UnicodeDecodeError:
        LOGGER.warning("Encoding-Fallback latin-1 für Datei %s.", path)
        return path.read_text(encoding="latin-1", errors="replace")


def _parse_timestamp_to_seconds(ts: str) -> float:
    """
    Erwartet Format HH:MM:SS.mmm
    """
    hh, mm, rest = ts.split(":")
    ss, ms = rest.split(".")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def _parse_vtt_to_segments(vtt_path: Path) -> List[VttCueSegment]:
    """
    Minimal-robustes WebVTT Parsing:
    - Ignoriert Header (WEBVTT) und Metadatenblöcke (NOTE/STYLE/REGION)
    - Erkennt Zeitzeilen: "HH:MM:SS.mmm --> HH:MM:SS.mmm"
    - Sammelt nachfolgende Textzeilen bis zur nächsten Leerzeile als Cue-Text
    """
    raw = _read_text_file_best_effort(vtt_path)
    lines = [ln.rstrip("\n").rstrip("\r") for ln in raw.splitlines()]

    segments: List[VttCueSegment] = []
    i = 0
    cue_index = 0

    def _skip_block(start_keyword: str, idx: int) -> int:
        # Skip lines until blank line
        j = idx
        while j < len(lines) and lines[j].strip() != "":
            j += 1
        # skip trailing blank lines
        while j < len(lines) and lines[j].strip() == "":
            j += 1
        return j

    # Skip initial WEBVTT header line if present
    if i < len(lines) and lines[i].strip().startswith("WEBVTT"):
        i += 1

    while i < len(lines):
        ln = lines[i].strip()

        if ln == "":
            i += 1
            continue

        # Skip NOTE/STYLE/REGION blocks
        if ln.startswith("NOTE"):
            i = _skip_block("NOTE", i + 1)
            continue
        if ln.startswith("STYLE"):
            i = _skip_block("STYLE", i + 1)
            continue
        if ln.startswith("REGION"):
            i = _skip_block("REGION", i + 1)
            continue

        # Optional cue identifier line: if next line is timing, current may be an ID
        timing_line = ln
        next_line = lines[i + 1].strip() if (i + 1) < len(lines) else ""
        if _TIME_RE.match(next_line):
            timing_line = next_line
            i += 1  # consume cue id line

        m = _TIME_RE.match(timing_line)
        if m:
            start_s = _parse_timestamp_to_seconds(m.group("start"))
            end_s = _parse_timestamp_to_seconds(m.group("end"))

            # Collect cue text lines until blank line
            i += 1
            text_lines: List[str] = []
            while i < len(lines) and lines[i].strip() != "":
                text_lines.append(lines[i].strip())
                i += 1

            cue_text = " ".join([t for t in text_lines if t]).strip()
            if cue_text:
                segments.append(
                    VttCueSegment(
                        cue_index=cue_index,
                        text=cue_text,
                        start=start_s,
                        end=end_s,
                    )
                )
                cue_index += 1
            continue

        # Fallback: keine Timingline erkannt -> behandle Zeile als "Plain text"
        # (robust für untypische/kaputte VTTs)
        plain = ln.strip()
        if plain:
            segments.append(VttCueSegment(cue_index=cue_index, text=plain, start=None, end=None))
            cue_index += 1
        i += 1

    # deterministisch
    return sorted(segments, key=lambda s: s.cue_index)

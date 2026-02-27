# src/semantic/models/extract_texts_models.py
"""
Datenmodelle für Schritt 1 (Textextraktion) der rein semantischen Pipeline.

Ziel:
- Output ist konsequent sessionweise strukturiert
- Optionales Segment/Cue-Level für spätere UEM/Time-Restriktion und Debuggability
- Deterministische Serialisierung (Sortierung)
- Eingebaute Validierung (Eindeutigkeit, Plausibilitäten)

Empfohlenes Output-JSON pro Session (Datei):
{
  "schema_version": 1,
  "session_name": "session_132",
  "speakers": [
    {
      "speaker_id": "spk_0",
      "uid": "session_132_spk_0",
      "text": "...",
      "segments": [
        {"cue_index": 0, "text": "...", "start": 12.34, "end": 15.67}
      ]
    }
  ]
}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


SCHEMA_VERSION: int = 1


@dataclass(frozen=True)
class VttCueSegment:
    """
    Repräsentiert einen VTT-Cue als Segment.

    start/end sind optional, um auch bei fehlerhaften/inkompletten VTTs robust zu bleiben.
    """
    cue_index: int
    text: str
    start: Optional[float] = None  # Sekunden
    end: Optional[float] = None    # Sekunden

    def __post_init__(self) -> None:
        if self.cue_index < 0:
            raise ValueError("VttCueSegment: 'cue_index' muss >= 0 sein.")
        if self.start is not None and self.start < 0:
            raise ValueError("VttCueSegment: 'start' muss >= 0 sein, sofern gesetzt.")
        if self.end is not None and self.end < 0:
            raise ValueError("VttCueSegment: 'end' muss >= 0 sein, sofern gesetzt.")
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("VttCueSegment: 'end' darf nicht kleiner als 'start' sein.")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cue_index": int(self.cue_index),
            "text": str(self.text),
            "start": self.start,
            "end": self.end,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VttCueSegment":
        return cls(
            cue_index=int(data["cue_index"]),
            text=str(data.get("text", "")),
            start=data.get("start", None),
            end=data.get("end", None),
        )


@dataclass(frozen=True)
class SpeakerExtractedText:
    """
    Extrahierter Text pro Speaker innerhalb einer Session.

    - speaker_id: z. B. "spk_0"
    - uid: stabiler globaler Identifier, z. B. "session_132_spk_0"
    - text: aggregierter Text über alle Segmente (kann leer sein)
    - segments: optionale Cue-Liste (für spätere Zeitrestriktionen/Debug)
    """
    speaker_id: str
    uid: str
    text: str
    segments: List[VttCueSegment] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.speaker_id:
            raise ValueError("SpeakerExtractedText: 'speaker_id' darf nicht leer sein.")
        if not self.uid:
            raise ValueError("SpeakerExtractedText: 'uid' darf nicht leer sein.")

        cue_indices = [s.cue_index for s in self.segments]
        if len(set(cue_indices)) != len(cue_indices):
            raise ValueError("SpeakerExtractedText: 'segments' enthält doppelte cue_index-Werte.")

    def to_dict(self) -> Dict[str, Any]:
        segments_sorted = sorted(self.segments, key=lambda s: s.cue_index)
        return {
            "speaker_id": str(self.speaker_id),
            "uid": str(self.uid),
            "text": str(self.text),
            "segments": [s.to_dict() for s in segments_sorted],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SpeakerExtractedText":
        segments_raw = data.get("segments", []) or []
        segments = [VttCueSegment.from_dict(seg) for seg in segments_raw]
        return cls(
            speaker_id=str(data.get("speaker_id", "")),
            uid=str(data.get("uid", "")),
            text=str(data.get("text", "")),
            segments=segments,
        )


@dataclass(frozen=True)
class SessionExtractedTexts:
    """
    Sessionweiser Container.

    Wird als einzelne JSON-Datei pro Session persistiert.
    """
    session_name: str
    speakers: List[SpeakerExtractedText]
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"SessionExtractedTexts: schema_version={self.schema_version} "
                f"wird nicht unterstützt (erwartet: {SCHEMA_VERSION})."
            )
        if not self.session_name:
            raise ValueError("SessionExtractedTexts: 'session_name' darf nicht leer sein.")

        speaker_ids = [s.speaker_id for s in self.speakers]
        if len(set(speaker_ids)) != len(speaker_ids):
            raise ValueError("SessionExtractedTexts: 'speaker_id' muss innerhalb der Session eindeutig sein.")

        uids = [s.uid for s in self.speakers]
        if len(set(uids)) != len(uids):
            raise ValueError("SessionExtractedTexts: 'uid' muss innerhalb der Session eindeutig sein.")

    def to_dict(self) -> Dict[str, Any]:
        speakers_sorted = sorted(self.speakers, key=lambda s: s.speaker_id)
        return {
            "schema_version": int(self.schema_version),
            "session_name": str(self.session_name),
            "speakers": [s.to_dict() for s in speakers_sorted],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SessionExtractedTexts":
        speakers_raw = data.get("speakers", []) or []
        speakers = [SpeakerExtractedText.from_dict(spk) for spk in speakers_raw]
        return cls(
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
            session_name=str(data.get("session_name", "")),
            speakers=speakers,
        )

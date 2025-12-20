"""
Module: segmentation_results
============================

Defines the data structures used to store the **output** of the active speaker 
segmentation process. These classes act as the blueprint for the results that 
are calculated by the algorithm.

Classes:
    - SpeakerSegment: A single time interval (start/end) of speech.
    - SpeakerActivity: A collection of segments for one specific speaker.
    - SegmentationResult: The top-level container for a full session's results.
"""

import json
import os
from dataclasses import dataclass, asdict


@dataclass
class SpeakerSegment:
    """
    Represents a single atomic time interval where a speaker is detected as active.
    This is the fundamental unit of the segmentation result.

    Attributes:
        start (float): The start time of the speech segment in seconds (e.g., 12.5).
        end (float): The end time of the speech segment in seconds (e.g., 14.2).
    """
    start: float
    end: float

    @property
    def duration(self) -> float:
        """Calculates the duration of the segment in seconds."""
        return self.end - self.start

    @classmethod
    def from_frames(cls, start_frame: int, end_frame: int, fps: float = 25.0):
        """
        Factory method to create a segment directly from frame indices.

        Args:
            start_frame (int): The inclusive start frame index.
            end_frame (int): The exclusive end frame index (Python slice style).
            fps (float, optional): Frames per second of the video source. Defaults to 25.0.

        Returns:
            SpeakerSegment: A new instance with calculated time values.
        """
        return cls(
            start=float(start_frame) / fps,
            end=float(end_frame) / fps
        )


@dataclass
class SpeakerActivity:
    """
    Aggregates ALL active segments for ONE specific speaker within a session.
    This represents the complete timeline of a speaker's activity.

    Attributes:
        speaker_id (str): The unique identifier of the speaker (e.g., 'spk_0').
        segments (list[SpeakerSegment]): A chronologically sorted list of active segments.
    """
    speaker_id: str
    segments: list[SpeakerSegment]

    def get_total_duration(self) -> float:
        """Returns the accumulated speaking time (sum of all segments) in seconds."""
        return sum(s.duration for s in self.segments)

    def add_segment_from_frames(self, start_f: int, end_f: int):
        """
        Helper method to append a new segment defined by frame indices using default FPS.

        Args:
            start_f (int): Start frame index.
            end_f (int): End frame index.
        """
        new_seg = SpeakerSegment.from_frames(start_f, end_f)
        self.segments.append(new_seg)


@dataclass
class SegmentationResult:
    """
    The complete calculated result container for one session.
    It maps speakers to their activity timelines and links the result back to 
    the input session and the configuration used.

    Attributes:
        session_id (str): The unique session identifier (e.g., 'session_40').
        experiment_id (str): The 8-char MD5 hash of the config used (e.g., 'a1b2c3d4').
        speakers (dict[str, SpeakerActivity]): The results organized by speaker.
            Key: speaker_id (str) - e.g., "spk_0"
            Value: SpeakerActivity object containing all segments for that speaker.
    """
    session_id: str
    experiment_id: str
    speakers: dict[str, SpeakerActivity]

    def get_speaker(self, speaker_id: str) -> SpeakerActivity:
        """Retrieves the activity object for a specific speaker."""
        return self.speakers.get(speaker_id)

    def save_to_disk(self, output_folder: str) -> str:
        """
        Saves the structured result to a JSON file.
        Format: <output_folder>/<session_id>.json

        Args:
            output_folder (str): The directory where the JSON file should be saved.

        Returns:
            str: The full file path of the saved JSON.
        """
        # Build a clean dictionary structure for JSON serialization
        data = {
            "session_id": self.session_id,
            "experiment_id": self.experiment_id,
            "speakers": {}
        }

        # Serialize speaker segments (nested objects -> list of dicts)
        for spk_id, activity in self.speakers.items():
            # asdict converts the Dataclass (Segment) to a simple dict
            data["speakers"][spk_id] = [asdict(seg) for seg in activity.segments]

        filename = f"{self.session_id}.json"
        file_path = os.path.join(output_folder, filename)

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        return file_path

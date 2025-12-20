"""
Module: session_input
=====================

Defines the data structures that act as the **input** for the pipeline.
It handles the representation of raw session data, including metadata and 
stitched Active Speaker Detection (ASD) scores per speaker.

These classes are used to load data from the disk into memory before any 
processing or segmentation algorithms are applied.

Classes:
    - SessionSpeakerData: Raw data (scores + UEM) for a single speaker.
    - SessionModel: Top-level container for a full session (metadata + all speakers).
"""

from dataclasses import dataclass


@dataclass
class SessionSpeakerData:
    """
    Holds the RAW input data for one speaker in a session.
    This data originates from the metadata.json and the stitched ASD files.

    Attributes:
        speaker_id (str): The unique identifier of the speaker (e.g., 'spk_0').
        uem_start (float): The start time of the valid evaluation map in seconds.
                           Data before this timestamp is considered invalid/noise.
        uem_end (float): The end time of the valid evaluation map in seconds.
                         Data after this timestamp is considered invalid/noise.
        asd_scores (dict[int, float]): A mapping of absolute frame numbers to ASD scores.
            Key: Frame number (int).
            Value: Active Speaker Detection score (float). Higher values indicate higher probability of speech.
    """
    speaker_id: str
    uem_start: float
    uem_end: float
    asd_scores: dict[int, float]

    def get_score(self, frame_idx: int) -> float:
        """
        Safe retrieval of an ASD score for a specific frame.

        Args:
            frame_idx (int): The absolute frame number to look up.

        Returns:
            float: The ASD score, or None if the frame index does not exist 
                   (e.g., gap in tracking or outside of video range).
        """
        return self.asd_scores.get(frame_idx, None)


@dataclass
class SessionModel:
    """
    Represents a complete recording session comprising multiple speakers.
    This acts as the primary Input Container for the processing pipeline.

    Attributes:
        session_id (str): The unique session identifier (e.g., 'session_40').
        folder_path (str): The absolute file system path to the session's raw data folder.
        metadata (dict): The complete raw content of the session's 'metadata.json'.
                         Kept for reference to allow access to auxiliary info if needed.
        input_data (dict[str, SessionSpeakerData]): The processed and structured data per speaker.
            Key: speaker_id (str) - e.g., "spk_0".
            Value: SessionSpeakerData object containing the stitched ASD scores and UEM limits.
    """
    session_id: str
    folder_path: str
    metadata: dict
    input_data: dict[str, SessionSpeakerData]

    def __repr__(self):
        """Returns a string representation of the session and speaker count."""
        return f"<SessionModel id='{self.session_id}' speakers={len(self.input_data)}>"
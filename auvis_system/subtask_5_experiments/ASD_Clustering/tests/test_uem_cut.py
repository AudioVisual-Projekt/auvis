"""
Test Module: UEM Cut Logic
==========================

This module tests the logic responsible for slicing Active Speaker Detection (ASD) scores
based on UEM (Un-Partitioned Evaluation Map) timestamps.

Focus:
    - Verifies the "Strict Inside" strategy.
    - Ensures frames starting before UEM start are excluded.
    - Ensures frames starting after UEM end are excluded.
    - Tests fractional timing (off-grid timestamps) to ensure precise rounding.
"""

import unittest
from auvis_system.subtask_5_experiments.ASD_Clustering.src.models.session_input import SessionModel, SessionSpeakerData
from auvis_system.subtask_5_experiments.ASD_Clustering.src.services.asd_scores_preprocess_service import _apply_uem_cut

# Constant assumption: The system operates at 25 Frames Per Second.
FPS = 25.0


class TestUemCutLogic(unittest.TestCase):
    """
    Test suite for the UEM cut logic.
    Inherits from unittest.TestCase (similar to JUnit).
    """

    def _create_mock_session(self, uem_start_sec: float, uem_end_sec: float, frames_indices: list) -> SessionModel:
        """
        Helper method to create a synthetic SessionModel in memory.

        Args:
            uem_start_sec (float): The start time of the valid segment.
            uem_end_sec (float): The end time of the valid segment.
            frames_indices (list): A list of integer frame indices to populate the session with.

        Returns:
            SessionModel: A populated session object ready for testing.
        """
        # Create dummy scores (value 1.0) for the requested frame indices
        scores = {f: 1.0 for f in frames_indices}

        speaker = SessionSpeakerData(
            speaker_id="test_spk",
            uem_start=uem_start_sec,
            uem_end=uem_end_sec,
            asd_scores=scores
        )

        # Wrap it in a SessionModel
        session = SessionModel(
            session_id="test_session",
            folder_path="/tmp",  # Dummy path, not used in this test
            metadata={},
            input_data={"test_spk": speaker}
        )

        return session

    def test_uem_fractional_cut_logic(self):
        """
        Tests the 'Strict Inside' logic using fractional timestamps.

        Scenario:
           The session ends exactly when a "whistle" blows at 2.5 seconds.
           We must ensure we do not include the frame that starts right after the whistle.

        Parameters:
           FPS:   25
           Start: 1.5 seconds
           End:   2.5 seconds

        Calculation:
           Start: 1.5 * 25 = 37.5 Frames 
                  -> Strategy: Ceil (Round Up) -> Frame 38 is the first valid frame.

           End:   2.5 * 25 = 62.5 Frames 
                  -> Strategy: Floor (Round Down) -> Frame 62 is the last valid frame.
        """
        start_time = 1.5
        end_time = 2.5

        # We create frames around the calculated boundaries to verify the logic:
        # Frame 37 (Starts at 1.48s) -> Must be EXCLUDED (Too early)
        # Frame 38 (Starts at 1.52s) -> Must be INCLUDED (First valid frame)
        # Frame 62 (Starts at 2.48s) -> Must be INCLUDED (Last valid frame)
        # Frame 63 (Starts at 2.52s) -> Must be EXCLUDED (Starts after end time)
        input_frames = [37, 38, 50, 62, 63]

        # 1. Setup
        session = self._create_mock_session(start_time, end_time, input_frames)

        # 2. Action
        # We pass None as logger because we don't need logging output in the test
        processed = _apply_uem_cut(session, None)

        # 3. Assertion
        result_keys = processed.input_data["test_spk"].asd_scores.keys()

        # Debug output (only visible if test fails or usually hidden in standard unittest)
        print(f"\nUEM Interval: {start_time}s to {end_time}s")
        print(f"Frames kept: {sorted(list(result_keys))}")

        # Check Start Boundary
        self.assertNotIn(37, result_keys, "Frame 37 starts before 1.5s -> Should be removed.")
        self.assertIn(38, result_keys, "Frame 38 starts after 1.5s  -> Should be kept.")

        # Check End Boundary
        self.assertIn(62, result_keys, "Frame 62 starts before 2.5s -> Should be kept.")
        self.assertNotIn(63, result_keys, "Frame 63 starts after 2.5s  -> Should be removed.")


if __name__ == '__main__':
    unittest.main()

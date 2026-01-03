import math
from typing import Dict, List, Any


class SegmentationAlgorithm:
    """
    Implements the core ASD segmentation logic based on the legacy 'segmentation.py'
    and 'conv_spks.py'.

    Workflow:
    1.  Sort raw frame scores.
    2.  Pass 1: Hysteresis Thresholding (Onset/Offset) -> Raw Regions.
    3.  Pass 2: Merging (Fill short gaps).
    4.  Pass 3: Splitting & Dropping (Min duration, Max chunk size).
    5.  Time Conversion (Absolute Seconds, NO shift applied for storage).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the algorithm with parameters from the config.
        Defaults are taken from the legacy 'CENTRAL_ASD_CHUNKING_PARAMETERS'.
        """
        self.onset = config.get("onset", 1.0)
        self.offset = config.get("offset", 0.8)

        # Duration parameters in seconds (will be converted to frames)
        self.min_duration_on_sec = config.get("min_duration_on", 1.0)
        self.min_duration_off_sec = config.get("min_duration_off", 0.5)
        self.max_chunk_size_sec = config.get("max_chunk_size", 10.0)
        self.min_chunk_size_sec = config.get("min_chunk_size", 1.0)

        # FPS fixed at 25 according to legacy code
        self.fps = 25

    def run(self, frames_dict: Dict[str, float]) -> List[List[float]]:
        """
        Runs the segmentation logic.

        Args:
            frames_dict: Dictionary { "frame_id_str": score_float, ... }

        Returns:
            List of segments in ABSOLUTE seconds (aligned with video time).
            Format: [[start_sec, end_sec], ...]
        """
        # 0. Pre-Calculation: Convert seconds parameters to frames
        min_duration_on_frames = int(self.min_duration_on_sec * self.fps)
        min_duration_off_frames = int(self.min_duration_off_sec * self.fps)
        max_chunk_frames = int(self.max_chunk_size_sec * self.fps)
        min_chunk_frames = int(self.min_chunk_size_sec * self.fps)

        # 1. Prepare Frames: Sort and convert to int
        # frames_dict keys are strings like "1454", convert to int
        frames = sorted([int(f) for f in frames_dict.keys()])

        if not frames:
            return []

        # Find min_frame for normalization (legacy logic behavior needed for merging steps)
        min_frame = frames[0]

        # --- PASS 1: Hysteresis Thresholding ---
        speech_regions = []
        current_region = None
        is_active = False

        for frame in frames:
            score = frames_dict.get(frame, frames_dict.get(str(frame), -1.0))

            # Normalize frame index (legacy behavior)
            normalized_frame = frame - min_frame

            if not is_active:
                # Check for onset
                if score > self.onset:
                    is_active = True
                    current_region = [normalized_frame]
            else:
                # Check for offset
                if score < self.offset:
                    is_active = False
                    if current_region is not None:
                        speech_regions.append(current_region)
                        current_region = None
                else:
                    # Still active or above offset threshold
                    current_region.append(normalized_frame)

        # Handle active region at the end
        if current_region is not None:
            speech_regions.append(current_region)

        # --- PASS 2: Merging (Fill Gaps) ---
        merged_regions = []
        if speech_regions:
            current_region = speech_regions[0]

            for next_region in speech_regions[1:]:
                # Calculate gap between end of current and start of next
                gap = next_region[0] - current_region[-1] - 1

                if gap <= min_duration_off_frames:
                    # Merge: Extend current region with next region
                    current_region.extend(next_region)
                else:
                    merged_regions.append(current_region)
                    current_region = next_region
            merged_regions.append(current_region)

        # --- PASS 3: Splitting & Dropping ---
        final_frame_segments = []
        for region in merged_regions:
            region_length = len(region)

            # Drop short regions
            if region_length < min_duration_on_frames:
                continue

            # Split long regions
            if region_length > max_chunk_frames:
                num_chunks = math.ceil(region_length / max_chunk_frames)
                chunk_size = math.ceil(region_length / num_chunks)

                for i in range(0, region_length, chunk_size):
                    sub_segment = region[i: i + chunk_size]
                    if len(sub_segment) >= min_chunk_frames:
                        final_frame_segments.append(sub_segment)
            else:
                final_frame_segments.append(region)

        # --- FINAL STEP: Convert to ABSOLUTE Seconds ---
        final_time_segments = []

        for segment in final_frame_segments:
            # Denormalize: Add min_frame back to get absolute frame ID
            start_frame_abs = segment[0] + min_frame
            end_frame_abs = segment[-1] + min_frame

            # Convert to absolute seconds (Frame ID / FPS)
            # Legacy Note: Using end_frame_abs / fps aligns with 'conv_spks.py'.
            # Technically this marks the START of the last frame, not its duration end.
            # But we keep it for 1:1 legacy compatibility.
            t_start_abs = start_frame_abs / self.fps
            t_end_abs = end_frame_abs / self.fps

            # NO SHIFT APPLIED (- uem_start removed)
            final_time_segments.append([t_start_abs, t_end_abs])

        return final_time_segments

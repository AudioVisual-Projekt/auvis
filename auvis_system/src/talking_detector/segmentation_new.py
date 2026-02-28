import math
import numpy as np

CENTRAL_ASD_CHUNKING_PARAMETERS = {
    "onset": 1.0,        # start threshold
    "offset": 0.8,       # end threshold
    "min_duration_on": 1.0,  # drop
    "min_duration_off": 0.5, # fill
    "max_chunk_size": 10,
    "min_chunk_size": 1
}

EGO_ASD_CHUNKING_PARAMETERS = {
    "onset": 2.4,        # start threshold
    "offset": 1.6,       # end threshold
    "min_duration_on": 1.0,  # drop
    "min_duration_off": 0.5, # fill
    "max_chunk_size": 10,
    "min_chunk_size": 1
}


def segment_by_asd(asd, parameters={}):
    onset_threshold = parameters.get("onset", CENTRAL_ASD_CHUNKING_PARAMETERS["onset"])
    offset_threshold = parameters.get("offset", CENTRAL_ASD_CHUNKING_PARAMETERS["offset"])
    
    # Convert frame numbers to integers and sort them
    frames = sorted([int(f) for f in asd.keys()])
    if not frames:
        return []
        
    # Find the minimum frame number to normalize frame indices
    min_frame = min(frames)
    
    # Convert duration parameters from seconds to frames (assuming 25 fps)
    min_duration_on_frames = int(parameters.get("min_duration_on",  CENTRAL_ASD_CHUNKING_PARAMETERS["min_duration_on"]) * 25)
    min_duration_off_frames = int(parameters.get("min_duration_off", CENTRAL_ASD_CHUNKING_PARAMETERS["min_duration_on"]) * 25)
    max_chunk_frames = int(parameters.get("max_chunk_size", CENTRAL_ASD_CHUNKING_PARAMETERS["max_chunk_size"]) * 25)
    min_chunk_frames = int(parameters.get("min_chunk_size", CENTRAL_ASD_CHUNKING_PARAMETERS["min_chunk_size"]) * 25)
    
    # First pass: Find speech regions using hysteresis thresholding
    speech_regions = []
    current_region = None
    is_active = False
    
    for frame in frames:
        score = asd.get(str(frame), -1)
        normalized_frame = frame - min_frame
        
        if not is_active:
            # Currently inactive, check for onset
            if score > onset_threshold:
                is_active = True
                current_region = [normalized_frame]
        else:
            # Currently active, check for offset
            if score < offset_threshold:
                is_active = False
                if current_region is not None:
                    speech_regions.append(current_region)
                    current_region = None
            else:
                current_region.append(normalized_frame)
    
    # Handle case where speech continues until the end
    if current_region is not None:
        speech_regions.append(current_region)
    
    # Second pass: Merge regions separated by short non-speech gaps
    merged_regions = []
    if speech_regions:
        current_region = speech_regions[0]
        
        for next_region in speech_regions[1:]:
            gap = next_region[0] - current_region[-1] - 1
            if gap <= min_duration_off_frames:
                # Merge regions
                current_region.extend(next_region)
            else:
                merged_regions.append(current_region)
                current_region = next_region
        merged_regions.append(current_region)
    
    # Third pass: Remove short speech regions and split long ones
    final_segments = []
    for region in merged_regions:
        region_length = len(region)
        
        # Skip regions shorter than minimum duration
        if region_length < min_duration_on_frames:
            continue
            
        # Split long regions
        if region_length > max_chunk_frames:
            num_chunks = math.ceil(region_length / max_chunk_frames)
            chunk_size = math.ceil(region_length / num_chunks)
            
            for i in range(0, region_length, chunk_size):
                sub_segment = region[i:i + chunk_size]
                if len(sub_segment) >= min_chunk_frames:
                    final_segments.append(sub_segment)
        else:
            final_segments.append(region)
    
    # Convert frame indices back to original frame indices
    final_segments = [
        [frame + min_frame for frame in segment]
        for segment in final_segments
    ]
    
    return final_segments




def segment_by_asd_new(asd, parameters={}, fps=25.0, duration_frames=None):
    frames = sorted([int(f) for f in asd.keys()])
    if not frames: return []

    # --- durations in frames ---
    min_on  = int(parameters.get("min_duration_on",  1.0) * fps)
    min_off = int(parameters.get("min_duration_off", 0.5) * fps)   # FIXED BUG
    max_len = int(parameters.get("max_chunk_size",   10)  * fps)
    min_len = int(parameters.get("min_chunk_size",   1)   * fps)

    # --- conditioning ---
    scores = np.array([asd[str(f)] for f in frames], dtype=float)
    # smooth
    if parameters.get("smooth_win", 7) > 1:
        win = int(parameters.get("smooth_win", 7))
        pad = win // 2
        xpad = np.pad(scores, (pad,pad), mode="edge")
        ker = np.ones(win)/win
        scores = np.convolve(xpad, ker, mode="valid")

    # --- thresholds (adaptive or fixed) ---
    if parameters.get("adaptive", True):
        onset, offset = adaptive_thresholds(scores, onset_p=parameters.get("onset_p",85),
                                                     offset_p=parameters.get("offset_p",60))
    else:
        onset  = parameters.get("onset",  1.0)
        offset = parameters.get("offset", 0.8)

    # --- hysteresis over frames ---
    speech_regions = []
    is_active, start_idx = False, None
    for i, f in enumerate(frames):
        s = scores[i]
        if not is_active and s >= onset:
            is_active, start_idx = True, i
        elif is_active and s <= offset:
            is_active = False
            speech_regions.append([start_idx, i])
    if is_active:
        speech_regions.append([start_idx, len(frames)-1])

    # --- merge short gaps ---
    merged = []
    if speech_regions:
        cur = speech_regions[0]
        for nxt in speech_regions[1:]:
            gap = nxt[0] - cur[1] - 1
            if gap <= min_off:
                cur[1] = nxt[1]
            else:
                merged.append(cur); cur = nxt
        merged.append(cur)

    # --- drop short, split long on valleys ---
    final_idx = []
    for a,b in merged:
        if (b-a+1) < min_on: continue
        if (b-a+1) > max_len:
            # valley-based split
            sub = split_on_valleys(list(range(a,b+1)), scores, max_len)
            for chunk in sub:
                if len(chunk) >= min_len: final_idx.append([chunk[0], chunk[-1]])
        else:
            final_idx.append([a,b])

    # --- padding & mapping back to original frames ---
    pad_pre = int(parameters.get("pad_pre", 0.10) * fps)
    pad_post= int(parameters.get("pad_post",0.15) * fps)
    min_frame = frames[0]
    out = []
    for a,b in final_idx:
        start_f = max(0, frames[a] - pad_pre)
        end_f   = frames[b] + pad_post
        if duration_frames is not None:
            end_f = min(end_f, duration_frames-1)
        if (end_f - start_f + 1) >= min_on:
            out.append([start_f, end_f])

    # QC
    total_cov = sum((e-s+1) for s,e in out)/fps if out else 0.0
    print(f"[seg] raw={len(speech_regions)} merge→{len(merged)} final={len(out)} "
          f"coverage_s={total_cov:.2f} onset={onset:.2f} offset={offset:.2f}")

    return out
 


import math


CENTRAL_ASD_CHUNKING_PARAMETERS = {
    "onset": 1.0,  # start threshold
    "offset": 0.8,  # end threshold
    "min_duration_on": 1.0,  # drop => 25 frames if 1.0 (see below)
    "min_duration_off": 0.5,  # fill => 12 frames if 0.5 (see below)
    "max_chunk_size": 10,  # 250 frames if 10
    "min_chunk_size": 1  # 25 frames if 1
}


def segment_by_asd(asd, parameters={}):
    # use parameters for onset and offset that are defined at the top
    onset_threshold = parameters.get("onset", CENTRAL_ASD_CHUNKING_PARAMETERS["onset"])
    offset_threshold = parameters.get("offset", CENTRAL_ASD_CHUNKING_PARAMETERS["offset"])
    
    # convert frame numbers to integers and sort them
    frames = sorted([int(f) for f in asd.keys()])
    if not frames:
        return []
        
    # find the minimum frame number to normalize frame indices
    min_frame = min(frames)
    
    # convert duration parameters from seconds to frames (assuming 25 fps)
    min_duration_on_frames = int(parameters.get("min_duration_on",  CENTRAL_ASD_CHUNKING_PARAMETERS["min_duration_on"]) * 25)
    min_duration_off_frames = int(parameters.get("min_duration_off", CENTRAL_ASD_CHUNKING_PARAMETERS["min_duration_off"]) * 25)
    max_chunk_frames = int(parameters.get("max_chunk_size", CENTRAL_ASD_CHUNKING_PARAMETERS["max_chunk_size"]) * 25)
    min_chunk_frames = int(parameters.get("min_chunk_size", CENTRAL_ASD_CHUNKING_PARAMETERS["min_chunk_size"]) * 25)
    
    ### "FIRST PASS": find speech regions using hysteresis thresholding
    # speech_regions is a list of all connected speech segments
    # each element of the list is a list by itself, so, it contains all the frames where the speaker is actively speaking
    speech_regions = []
    current_region = None
    # initialized that the speaker is not talking at the moment (before analyzing first frame)
    is_active = False

    # all "normalized" frames one by one - comparison with onset and offset
    for frame in frames:
        score = asd.get(str(frame), -1)
        # normalized frame is built because it does not always start with frame 0
        normalized_frame = frame - min_frame
        
        if not is_active:
            # currently inactive, check for onset
            if score > onset_threshold:
                # if not active but score now bigger than threshold, set active and current (normalized) frame
                is_active = True
                # create list with this frame as the first (and only) entry
                current_region = [normalized_frame]
        else:
            # currently active, check for offset
            if score < offset_threshold:
                # set inactive, if score is too low and was active before
                is_active = False
                if current_region is not None:
                    # todo: How could it be possible that current_region is None? it should be set before, when it is active? which edge case is treated here?
                    speech_regions.append(current_region)
                    current_region = None
            # if already active and current score big enough
            else:
                # add this frame to the current_region list
                current_region.append(normalized_frame)
    
    # handle case where speech continues until the end
    if current_region is not None:
        speech_regions.append(current_region)
    
    ### "SECOND PASS": merge regions separated by short non-speech gaps
    merged_regions = []
    if speech_regions:
        # start with first region [0] then compare it with the next (in loop); go 1 by 1
        current_region = speech_regions[0]

        # compare two regions each
        # calculate distance from the 1st element of the 2nd segment to the last element of the 1st segment
        for next_region in speech_regions[1:]:
            gap = next_region[0] - current_region[-1] - 1
            # gap smaller than defined min_duration_off_frames - it is a number of frames (not seconds anymore!)
            if gap <= min_duration_off_frames:
                # Merge regions
                current_region.extend(next_region)
            else:
                merged_regions.append(current_region)
                current_region = next_region
        merged_regions.append(current_region)
    
    # "THIRD PASS": remove short speech regions and split long ones
    # todo: now the long ones (maybe merged ones) will be separated again?
    final_segments = []
    for region in merged_regions:
        region_length = len(region)
        
        # skip regions shorter than minimum duration => they will not be part of final_segments!
        if region_length < min_duration_on_frames:
            continue
            
        # split long regions
        # todo: how exactly is it splitted?
            # todo: what happens to regions that only slighter have more frames than the maximum?
                # answer: 300 frames => 150/150
        if region_length > max_chunk_frames:
            num_chunks = math.ceil(region_length / max_chunk_frames)
            chunk_size = math.ceil(region_length / num_chunks)
            
            for i in range(0, region_length, chunk_size):
                sub_segment = region[i:i + chunk_size]
                # if one of the new segments is too small after split, then it is thrown away
                if len(sub_segment) >= min_chunk_frames:
                    final_segments.append(sub_segment)
        else:
            final_segments.append(region)
    
    # convert frame indices (normalized_frames) back to original frame indices
    final_segments = [
        [frame + min_frame for frame in segment]
        for segment in final_segments
    ]
    
    return final_segments

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from typing import List, Tuple, Dict
from sklearn.cluster import AgglomerativeClustering
import json
import glob
from team_c.src.cluster.eval import pairwise_f1_score, pairwise_f1_score_per_speaker  # changed absolute path
from team_c.src.talking_detector.segmentation import segment_by_asd  # changed absolute path
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm
import traceback


def calculate_overlap_duration(segments1: List[Tuple[float, float]],
                             segments2: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Calculate total overlap and non-overlap duration between two speakers' segments.
    
    Args:
        segments1: List of (start, end) tuples for first speaker
        segments2: List of (start, end) tuples for second speaker
        
    Returns:
        Tuple of (total_overlap_duration, total_non_overlap_duration)
    """
    total_overlap = 0.0
    total_non_overlap = 0.0
    
    # calculate total duration of each speaker's segments (in seconds)
    total_duration1 = sum(end - start for start, end in segments1)
    total_duration2 = sum(end - start for start, end in segments2)
    
    # calculate overlaps - simple inner and outer loop over all combinations of segments from speaker 1 and 2
    for start1, end1 in segments1:
        for start2, end2 in segments2:
            # Calculate overlap
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
            if overlap_end > overlap_start:
                total_overlap += overlap_end - overlap_start
    
    # calculate non-overlap
    total_non_overlap = total_duration1 + total_duration2 - 2 * total_overlap
    
    return total_overlap, total_non_overlap

def calculate_conversation_scores(speaker_segments: Dict[str, List[Tuple[float, float]]]) -> np.ndarray:
    """
    Calculate conversation likelihood scores between all pairs of speakers.
    Higher score means speakers are more likely to be in the same conversation.
    
    Args:
        speaker_segments: Dictionary mapping speaker IDs to their time segments
        
    Returns:
        NxN numpy array of conversation scores
    """
    n_speakers = len(speaker_segments)
    scores = np.zeros((n_speakers, n_speakers))
    speaker_ids = list(speaker_segments.keys())
    
    for i in range(n_speakers):
        for j in range(i + 1, n_speakers):
            spk1 = speaker_ids[i]
            spk2 = speaker_ids[j]
            
            overlap, non_overlap = calculate_overlap_duration(
                speaker_segments[spk1],
                speaker_segments[spk2]
            )
            
            # calculate conversation likelihood score
            # higher score when there's less overlap (more likely to be in same conversation)
            if overlap + non_overlap > 0:  # todo: only possible if both speakers does not talk whole time?
                # normalize overlap by total duration to get overlap ratio
                # percentage which part of the total_duration is overlap
                total_duration = overlap + non_overlap
                overlap_ratio = overlap / total_duration
                # convert to conversation likelihood (1 - overlap_ratio)
                score = 1 - overlap_ratio
            else:
                score = 0
                
            scores[i, j] = score
            scores[j, i] = score  # symmetric
    
    return scores

def cluster_speakers(scores: np.ndarray, 
                    speaker_ids: List[str],
                    threshold: float = 0.7) -> Dict[int, List[str]]:
    """
    Cluster speakers based on their conversation scores.
    
    Args:
        scores: NxN numpy array of conversation scores
        speaker_ids: list of speaker IDs
        threshold: minimum score to consider speakers in same conversation
        
    Returns:
        Dictionary mapping cluster IDs to lists of speaker IDs
    """
    # convert scores to distance matrix (1 - score) - so, like before with the overlap percentage
    distances = 1 - scores
    
    # perform hierarchical clustering: use threshold to determine number of clusters
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1-threshold,
        metric='precomputed',
        linkage='complete'
    )
    
    cluster_labels = clustering.fit_predict(distances)
    
    # group speakers by cluster
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(speaker_ids[i])
    
    # speaker to cluster mapping
    spk_to_cluster = {spk_id: label.item() for spk_id, label in zip(speaker_ids, cluster_labels)}
    return spk_to_cluster

def get_speaker_activity_segments(pycrop_asd_path: List[str], uem_start: int, uem_end: int) -> List[Tuple[float, float]]:
    # sort the list of paths to the track asd files for the speaker
    pycrop_asd_path = sorted(pycrop_asd_path)
    all_frames = dict({})

    # loop over each track_**_asd.json file of the speaker
    # todo: find out why there are multiple asd files for some speakers within one session
    for asd_path in pycrop_asd_path:
        with open(asd_path, "r") as f:
            asd_data = json.load(f)
            # fill with frame_number (key) and asd score (value) for ALL asd files in ONE dictionary
            all_frames.update(asd_data)

    # convert dictionary "all_frames" (key=frame, value=asd_value) to list of tuples "sorted_frames" [(frame, asd_value),...]
    sorted_frames = sorted(all_frames.items(), key=lambda x: int(x[0]))

    # to this point was only transforming of the track asd data, now the segmentation will happen
    # todo: integrate a possibility to get back the number of frames that exceeded the max_chunk_frame_length
    activity_segments = segment_by_asd(all_frames)

    # convert frame_number into seconds: 1 second = 25 frames
    activity_segments = [
        (int(segment[0])/25, int(segment[-1])/25)
        for segment in activity_segments
    ]
    
    aligned_activity_segments = []
    for segment in activity_segments:
        # todo: it is allowed that the segment starts before the official beginning (segment[0])?
            # todo: find out if the result includes really regions that start before uem_start
        if segment[1] < uem_start:
            continue
        if segment[0] > uem_end:
            break
        aligned_activity_segments.append([
            # max(segment[0] - uem_start, 0),
            # min(segment[1] - uem_start, uem_end - uem_start)
            segment[0] - uem_start,
            segment[1] - uem_start
        ])
    
    return aligned_activity_segments

def get_clustering_f1_score(conversation_clusters_label: Dict[str, int], true_clusters_label: Dict[str, int]) -> float:
    """
    Calculate F1 score for clustering results.
    
    Args:
        conversation_clusters_label: Dictionary mapping speaker IDs to cluster IDs
        true_clusters_label: Dictionary mapping speaker IDs to true cluster IDs
        
    Returns:
        F1 score
    """
    spk_list = list(conversation_clusters_label.keys())
    true_clusters = [true_clusters_label[spk] for spk in spk_list]
    conversation_clusters = [conversation_clusters_label[spk] for spk in spk_list]
    return pairwise_f1_score(true_clusters, conversation_clusters)

def get_speaker_clustering_f1_score(conversation_clusters_label: Dict[str, int], true_clusters_label: Dict[str, int]) -> Dict[str, float]:
    spk_list = list(conversation_clusters_label.keys())
    true_clusters = [true_clusters_label[spk] for spk in spk_list]
    conversation_clusters = [conversation_clusters_label[spk] for spk in spk_list]
    f1_scores = pairwise_f1_score_per_speaker(true_clusters, conversation_clusters)
    spk_to_f1_score = {spk: round(f1_scores[i], 4) for i, spk in enumerate(spk_list)}
    return spk_to_f1_score

def get_clustering_ari_score(conversation_clusters_label: Dict[str, int], true_clusters_label: Dict[str, int]) -> float:
    spk_list = list(conversation_clusters_label.keys())
    true_clusters = [true_clusters_label[spk] for spk in spk_list]
    conversation_clusters = [conversation_clusters_label[spk] for spk in spk_list]
    return adjusted_rand_score(true_clusters, conversation_clusters)
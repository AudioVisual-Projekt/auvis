import os
os.sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
import jiwer
import webvtt
import json
from src.cluster.conv_spks import (
    get_clustering_f1_score,
    get_speaker_clustering_f1_score
)
from src.tokenizer.norm_text import remove_disfluencies
import glob
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
text_normalizer = EnglishTextNormalizer({})

def evaluate_conversation_clustering(label_path, output_path):
    with open(os.path.join(label_path, "speaker_to_cluster.json"), "r") as f:
        label_data = json.load(f)
    with open(os.path.join(output_path, "speaker_to_cluster.json"), "r") as f:
        output_data = json.load(f)
    return get_clustering_f1_score(label_data, output_data)

def evaluate_speaker_clustering(label_path, output_path):
    with open(os.path.join(label_path, "speaker_to_cluster.json"), "r") as f:
        label_data = json.load(f)
    with open(os.path.join(output_path, "speaker_to_cluster.json"), "r") as f:
        output_data = json.load(f)
    return get_speaker_clustering_f1_score(label_data, output_data)


def benchmark_vtt_wer(ref_vtt, hypo_vtt, ref_uem_start, ref_uem_end, hypo_uem_start, hypo_uem_end, show_diff=False):
    ref_strings = []
    hypo_strings = []
    for caption in webvtt.read(ref_vtt):
        if caption.start_in_seconds + caption.start_time.milliseconds/1000 < ref_uem_start:
            continue
        if caption.end_in_seconds + caption.end_time.milliseconds/1000 > ref_uem_end:
            continue
        ref_strings.append(remove_disfluencies(text_normalizer(caption.text)))
    for caption in webvtt.read(hypo_vtt):
        if caption.start_in_seconds + caption.start_time.milliseconds/1000 < hypo_uem_start:
            continue
        if caption.end_in_seconds + caption.end_time.milliseconds/1000 > hypo_uem_end:
            continue
        hypo_strings.append(remove_disfluencies(text_normalizer(caption.text)))
    
    if show_diff:
        # Show the WER error type (insertion, deletion, substitution) using wer library
        out = jiwer.process_words(
            [" ".join(ref_strings)],
            [" ".join(hypo_strings)],
        )
        print(jiwer.visualize_alignment(out))

    return jiwer.wer(" ".join(ref_strings), " ".join(hypo_strings))

def evaluate_speaker_transcripts(label_path, output_path, speaker_list, speaker_uem_start, speaker_uem_end):
    speaker_to_wer = {}
    for speaker, uem_start, uem_end in zip(speaker_list, speaker_uem_start, speaker_uem_end):
        ref_vtt = os.path.join(label_path, f"{speaker}.vtt")
        hypo_vtt = os.path.join(output_path, f"{speaker}.vtt")
        wer_score = benchmark_vtt_wer(ref_vtt, hypo_vtt, uem_start, uem_end, uem_start, uem_end)
        speaker_to_wer[speaker] = round(wer_score, 4)
    return speaker_to_wer

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate speaker clustering and transcripts from video")
    parser.add_argument('--session_dir', type=str, required=True, help='Path to folder containing session data')
    parser.add_argument('--output_dir_name', type=str, default='output', help='Name of the output directory within each session (default: output)')
    parser.add_argument('--label_dir_name', type=str, default='labels', help='Name of the label directory within each session (default: labels)')
    opt = parser.parse_args()

    if opt.session_dir.strip().endswith("*"):
        all_session_dirs = glob.glob(opt.session_dir)
    else:
        all_session_dirs = [opt.session_dir]
    print(f"Evaluating {len(all_session_dirs)} sessions")

    all_conversation_clustering_f1_score = []
    all_speaker_wer = []
    all_cluster_speaker_wer = []

    for session_dir in all_session_dirs:
        print(f"Evaluating session {session_dir.split('/')[-1]}")
        label_path = os.path.join(session_dir, opt.label_dir_name)
        output_path = os.path.join(session_dir, opt.output_dir_name)
        assert os.path.exists(label_path), f"Label path {label_path} does not exist"
        assert os.path.exists(output_path), f"Output path {output_path} does not exist"
        
        with open(os.path.join(session_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
        speaker_list = list(metadata.keys())
        speaker_uem_start = [metadata[spk]['central']["uem"]["start"] for spk in speaker_list]
        speaker_uem_end = [metadata[spk]['central']["uem"]["end"] for spk in speaker_list]

        conversation_clustering_f1_score = evaluate_conversation_clustering(label_path, output_path)
        print(f"Conversation clustering F1 score: {conversation_clustering_f1_score}")
        all_conversation_clustering_f1_score.append(conversation_clustering_f1_score)


        speaker_to_wer = evaluate_speaker_transcripts(label_path, output_path, speaker_list, speaker_uem_start, speaker_uem_end)
        print(f"Speaker to WER: {speaker_to_wer}")
        all_speaker_wer.extend(list(speaker_to_wer.values()))

        speaker_clustering_f1_score = evaluate_speaker_clustering(label_path, output_path)
        print(f"Speaker clustering F1 score: {speaker_clustering_f1_score}")

        cluster_speaker_to_wer = {}
        for speaker, wer in speaker_to_wer.items():
            cluster_speaker_wer = 0.5 * wer + 0.5 * (1 - speaker_clustering_f1_score[speaker])
            cluster_speaker_to_wer[speaker] = cluster_speaker_wer
        print(f"Joint ASR-Clustering Error Rate: {cluster_speaker_to_wer}")
        all_cluster_speaker_wer.extend(list(cluster_speaker_to_wer.values()))

    print(f"Average Conversation Clustering F1 score: {sum(all_conversation_clustering_f1_score) / len(all_conversation_clustering_f1_score)}")
    print(f"Average Speaker WER: {sum(all_speaker_wer) / len(all_speaker_wer)}")
    print(f"Average Joint ASR-Clustering Error Rate: {sum(all_cluster_speaker_wer) / len(all_cluster_speaker_wer)}")

if __name__ == "__main__":
    main()
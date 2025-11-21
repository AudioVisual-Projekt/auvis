import os
os.sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__))))
import jiwer
import webvtt
import json
import glob
from collections import defaultdict

from src.cluster.conv_spks import (
    get_clustering_f1_score,
    get_speaker_clustering_f1_score
)
from src.tokenizer.norm_text import remove_disfluencies
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
        if caption.start_in_seconds + caption.start_time.milliseconds / 1000 < ref_uem_start:
            continue
        if caption.end_in_seconds + caption.end_time.milliseconds / 1000 > ref_uem_end:
            continue
        ref_strings.append(remove_disfluencies(text_normalizer(caption.text)))
    for caption in webvtt.read(hypo_vtt):
        if caption.start_in_seconds + caption.start_time.milliseconds / 1000 < hypo_uem_start:
            continue
        if caption.end_in_seconds + caption.end_time.milliseconds / 1000 > hypo_uem_end:
            continue
        hypo_strings.append(remove_disfluencies(text_normalizer(caption.text)))

    if show_diff:
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
    parser.add_argument('--session_dir', type=str, required=True,
                        help='Path to folder containing session data (supports glob patterns with *)')
    parser.add_argument('--output_dir_name', type=str, default='output',
                        help='Prefix of the output directory within each session (default: output). '
                             'All directories starting with this prefix will be evaluated separately.')
    parser.add_argument('--label_dir_name', type=str, default='labels',
                        help='Name of the label directory within each session (default: labels)')
    parser.add_argument('--seg_strat', type=str, default='core',
                        help='Segmentation Strategy (unused in evaluation, kept for compatibility)')
    opt = parser.parse_args()

    # Sessions sammeln
    if opt.session_dir.strip().endswith("*"):
        all_session_dirs = glob.glob(opt.session_dir)
    else:
        all_session_dirs = [opt.session_dir]
    print(f"Evaluating {len(all_session_dirs)} sessions")

    # Ergebnisse pro Output-Variante sammeln
    all_conversation_clustering_f1_score = defaultdict(list)
    all_speaker_wer = defaultdict(list)
    all_cluster_speaker_wer = defaultdict(list)

    for session_dir in all_session_dirs:
        session_name = session_dir.split('/')[-1]
        print(f"\n=== Evaluating session {session_name} ===")
        label_path = os.path.join(session_dir, opt.label_dir_name)
        assert os.path.exists(label_path), f"Label path {label_path} does not exist"

        with open(os.path.join(session_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
        speaker_list = list(metadata.keys())
        speaker_uem_start = [metadata[spk]['central']["uem"]["start"] for spk in speaker_list]
        speaker_uem_end = [metadata[spk]['central']["uem"]["end"] for spk in speaker_list]

        # Alle Output-Ordner finden, die mit output_dir_name anfangen (z.B. output, output_core, output_new)
        output_pattern = os.path.join(session_dir, opt.output_dir_name + "*")
        output_dirs = sorted(
            d for d in glob.glob(output_pattern)
            if os.path.isdir(d)
        )
        assert output_dirs, f"No output directories matching {output_pattern} found in {session_dir}"

        for output_path in output_dirs:
            variant_name = os.path.basename(output_path)
            print(f"\n--- Evaluating output dir: {variant_name} ---")
            assert os.path.exists(output_path), f"Output path {output_path} does not exist"

            # Conversation clustering
            conversation_clustering_f1_score = evaluate_conversation_clustering(label_path, output_path)
            print(f"Conversation clustering F1 score: {conversation_clustering_f1_score}")
            all_conversation_clustering_f1_score[variant_name].append(conversation_clustering_f1_score)

            # Speaker WER
            speaker_to_wer = evaluate_speaker_transcripts(
                label_path, output_path, speaker_list, speaker_uem_start, speaker_uem_end
            )
            print(f"Speaker to WER: {speaker_to_wer}")
            all_speaker_wer[variant_name].extend(list(speaker_to_wer.values()))

            # Speaker clustering
            speaker_clustering_f1_score = evaluate_speaker_clustering(label_path, output_path)
            print(f"Speaker clustering F1 score: {speaker_clustering_f1_score}")

            # Joint ASR-Clustering Error
            cluster_speaker_to_wer = {}
            for speaker, wer in speaker_to_wer.items():
                cluster_speaker_wer = 0.5 * wer + 0.5 * (1 - speaker_clustering_f1_score[speaker])
                cluster_speaker_to_wer[speaker] = cluster_speaker_wer
            print(f"Joint ASR-Clustering Error Rate: {cluster_speaker_to_wer}")
            all_cluster_speaker_wer[variant_name].extend(list(cluster_speaker_to_wer.values()))

    # Aggregierte Ergebnisse pro Variante ausgeben
    print("\n================= SUMMARY =================")
    for variant_name in sorted(all_conversation_clustering_f1_score.keys()):
        avg_conv_f1 = (
            sum(all_conversation_clustering_f1_score[variant_name]) /
            len(all_conversation_clustering_f1_score[variant_name])
        )
        avg_spk_wer = (
            sum(all_speaker_wer[variant_name]) /
            len(all_speaker_wer[variant_name])
            if all_speaker_wer[variant_name] else float('nan')
        )
        avg_cluster_err = (
            sum(all_cluster_speaker_wer[variant_name]) /
            len(all_cluster_speaker_wer[variant_name])
            if all_cluster_speaker_wer[variant_name] else float('nan')
        )

        print(f"\nResults for output dir prefix variant '{variant_name}':")
        print(f"  Average Conversation Clustering F1 score: {avg_conv_f1}")
        print(f"  Average Speaker WER: {avg_spk_wer}")
        print(f"  Average Joint ASR-Clustering Error Rate: {avg_cluster_err}")


if __name__ == "__main__":
    main()

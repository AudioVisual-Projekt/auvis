import os
import re
import csv
import pandas as pd


def time_to_frame(t: str) -> int:
    """
    Convert VTT timestamp 'HH:MM:SS.mmm' to frame number.
    """
    hh, mm, ss_ms = t.split(':')
    ss, ms = ss_ms.split('.')
    total_seconds = int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000
    return round(total_seconds * fps)


def parse_vtt_intervals(vtt_path: str):
    """
    Extract (start_frame, end_frame) intervals from a VTT file.
    """
    intervals = []
    timestamp_pattern = re.compile(
        r"(\d\d:\d\d:\d\d\.\d\d\d)\s-->\s(\d\d:\d\d:\d\d\.\d\d\d)"
    )

    with open(vtt_path, "r", encoding="utf-8") as f:
        for line in f:
            match = timestamp_pattern.search(line)
            if match:
                start, end = match.groups()
                start_f = time_to_frame(start)
                end_f = time_to_frame(end)
                intervals.append((start_f, end_f))

    return intervals


def convert_vtt_to_dataframe(vtt_path: str) -> pd.DataFrame:
    """
    Convert VTT file directly to pandas DataFrame
    with speaking frames labeled.
    """
    intervals = parse_vtt_intervals(vtt_path)

    frames = []
    for start_f, end_f in intervals:
        frames.extend(range(start_f, end_f + 1))

    df = pd.DataFrame({
        "frame": frames,
        "label": "SPEAKING",
        "label_id": 1
    })

    df = df.set_index("frame")

    return df


# ============================= MAIN ==============================

data_path = "data-bin/dev"
fps = 25

for session in os.listdir(data_path):

    if session != "session_42":
        continue

    print(session)

    for speaker in os.listdir(f"{data_path}/{session}/speakers"):

        if speaker != "spk_3":
            continue

        print(speaker)

        vtt_file = f"{data_path}/{session}/labels/{speaker}.vtt"
        vtt_df = convert_vtt_to_dataframe(vtt_file)

        files = os.listdir(f"{data_path}/{session}/speakers/{speaker}/central_crops")

        pattern = re.compile(r'^(track_\d{2})\.mp4$')
        
        tracks = [m.group(1) for f in files if (m := pattern.match(f))]

        for track in tracks:

            if track != "track_00":
                continue

            print(track)

            bbox_file = f"{data_path}/{session}/speakers/{speaker}/central_crops/{track}_bbox.json"
            bbox_df = pd.read_json(bbox_file).T
            bbox_df.rename(columns={"x1": "entity_box_x1", "y1": "entity_box_y1", "x2": "entity_box_x2", "y2": "entity_box_y2"}, inplace=True)
            bbox_df.index.name = "frame"

            df = pd.merge(vtt_df, bbox_df, how="right", on="frame")
            df.reset_index(inplace=True)
            
            # Frames in denen nicht gesprochen wird als solche labeln
            df["label"] = df["label"].fillna("NOT_SPEAKING")
            df["label_id"] = df["label_id"].fillna(0)
            df["label_id"] = df["label_id"].astype(int)
            
            df = df.drop_duplicates()
            df.reset_index(inplace=True, drop=True)
    
            asd_label = df[["frame","label_id"]].iloc[:-1]
            asd_label.rename(columns={"label_id": "label"}, inplace=True)
    
            asd_label.to_csv(f"{data_path}/{session}/labels/{speaker}_{track}.csv", index=False)
            print(f"Label gespeichert unter: {data_path}/{session}/labels/{speaker}_{track}.csv")

            break
        break
    break

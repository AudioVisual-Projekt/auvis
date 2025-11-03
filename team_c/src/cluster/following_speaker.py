import itertools
import numpy as np
import pandas as pd
import os
import pickle
import csv
from pprint import pprint
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering
from typing import Dict, List
import matplotlib.pyplot as plt
from team_c.src.cluster.eval import pairwise_f1_score, pairwise_f1_score_per_speaker  # changed absolute path
from team_c.script.main import inference
from team_c.src.cluster.new_score_calc import calculate_overlap_duration, calculate_conversation_scores


def merge_consecutive_segments_per_speakers(spk_segs):
    # In eine Liste von Dicts umwandeln
    entries = [
        {"speaker": spk, "start": start, "end": end}
        for spk, segs in spk_segs.items()
        for start, end in segs
    ]

    # DataFrame erstellen
    spk_segs_df = pd.DataFrame(entries)

    # Zeitfenster in Sekunden
    merge_threshold = 2.0

    # Ergebnisliste
    merged_segments = []

    # Pro Sprecher iterieren
    for spk, group in spk_segs_df.groupby("speaker"):
        # Segmente nach Startzeit sortieren
        group = group.sort_values("start").reset_index(drop=True)

        # Initialisiere erstes Segment
        current_start = group.loc[0, "start"]
        current_end = group.loc[0, "end"]

        # Durch alle weiteren Segmente iterieren
        for i in range(1, len(group)):
            row = group.loc[i]
            if row["start"] - current_end <= merge_threshold:
                # Segmente zusammenfassen
                current_end = max(current_end, row["end"])
            else:
                # Aktuelles Segment speichern und neues beginnen
                merged_segments.append({"speaker": spk, "start": current_start, "end": current_end})
                current_start = row["start"]
                current_end = row["end"]

        # Letztes Segment speichern
        merged_segments.append({"speaker": spk, "start": current_start, "end": current_end})

    # In DataFrame umwandeln
    merged_df = pd.DataFrame(merged_segments)
    return merged_df



if __name__ == "__main__":
    ############  Ziel ist es, herauszufinden, welcher Sprecher als nächstes beginnt, wenn ein Sprecher endet
    #### Hoffnung: Sprecher einer gemeinsamen Konversation wechseln sich ab

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inference_result_pkl_file = os.path.join(project_root,  "inference_result.pkl")
    if os.path.exists(inference_result_pkl_file):
        with open(inference_result_pkl_file, "rb") as f:
            inference_result = pickle.load(f)
    else:
        inference_result = inference()
        # # Ordner anlegen (falls nicht vorhanden)
        os.makedirs(os.path.dirname(inference_result_pkl_file), exist_ok=True)
        with open(inference_result_pkl_file, 'wb') as f:
            pickle.dump(inference_result, f, protocol=pickle.HIGHEST_PROTOCOL)

    spk_segs = inference_result['session_speaker_segments']
    # pprint(inference_result.iloc[0])
    # pprint(spk_segs.iloc[0])

### example:
my_segs = {
    'spk_0': [[1, 2], [3, 4]],
    'spk_1': [[5, 6], [9, 10]],
    'spk_2': [[7, 8], [11, 12]]
}
#
# # Flatten all segments with speaker info
# entries = []
# for spk, segs in my_segs.items():
#     for seg in segs:
#         start, end = seg
#         entries.append((start, end, spk))
#
# # Sort by start time
# entries.sort(key=lambda x: x[0])
#
# # Print with numbering
# for i, (start, end, spk) in enumerate(entries, 1):
#     print(f"{i}) {spk} [{start}, {end}]")
#
# import pandas as pd
#
# # Build and sort entries
# entries = [
#     {'speaker': spk, 'start': seg[0], 'end': seg[1]}
#     for spk, segs in my_segs.items()
#     for seg in segs
# ]
# df = pd.DataFrame(entries).sort_values(by='start').reset_index(drop=True)
#
# print(df)

#### alternativ, wahrscheinlich besser:

my_segs = {
    'spk_0': [[1, 2], [3, 4]],
    'spk_1': [[5, 6], [9, 10]],
    'spk_2': [[7, 8], [11, 12]]
}

my_segs = spk_segs

# # Alle Segmente in eine sortierte Liste bringen
# entries = []
# for spk, segs in my_segs.items():
#     for start, end in segs:
#         entries.append({'speaker': spk, 'start': start, 'end': end})
#
# # Nach Startzeit sortieren
# entries.sort(key=lambda x: x['start'])
#
# # Zeitfenster in Sekunden
# time_window = 2
#
# # Durch alle Segmente gehen und prüfen, ob nächster Sprecher innerhalb des Fensters beginnt
# results = []
# for i in range(len(entries) - 1):
#     current = entries[i]
#     next_seg = entries[i + 1]
#
#     gap = next_seg['start'] - current['end']
#     if 0 < gap <= time_window:
#         results.append({
#             'from_speaker': current['speaker'],
#             'to_speaker': next_seg['speaker'],
#             'gap': gap,
#             'from_end': current['end'],
#             'to_start': next_seg['start']
#         })
#
# # Ausgabe
# for r in results:
#     print(f"{r['from_speaker']} → {r['to_speaker']} (startet {r['gap']} s nach Ende, "
#           f"Ende={r['from_end']}, Start={r['to_start']})")

######################### Zählung der Sprecherwechsel
# from collections import Counter
#
# # Dein Dictionary mit Segmenten
# my_segs = spk_segs.iloc[0]
#
# # Alle Segmente sammeln
# entries = []
# for spk, segs in my_segs.items():
#     for start, end in segs:
#         entries.append({'speaker': spk, 'start': start, 'end': end})
#
# # Nach Startzeit sortieren
# entries.sort(key=lambda x: x['start'])
#
# # Zeitfenster in Sekunden
# time_window = 2.0
#
# # Ergebnisse und Wechselzählung
# results = []
# switch_counts = Counter()
#
# for i in range(len(entries) - 1):
#     current = entries[i]
#     next_seg = entries[i + 1]
#
#     gap = next_seg['start'] - current['end']
#     if 0 < gap <= time_window:
#         results.append({
#             'from_speaker': current['speaker'],
#             'to_speaker': next_seg['speaker'],
#             'gap': round(gap, 3),
#             'from_end': round(current['end'], 3),
#             'to_start': round(next_seg['start'], 3)
#         })
#         switch_counts[(current['speaker'], next_seg['speaker'])] += 1
#
# # Einzelne Übergänge ausgeben
# for r in results:
#     print(f"{r['from_speaker']} → {r['to_speaker']} "
#           f"(startet {r['gap']} s nach Ende, Ende={r['from_end']}, Start={r['to_start']})")
#
# print("\n--- Wechselzählung ---")
# # Aggregierte Wechsel zählen
# for (from_spk, to_spk), count in switch_counts.items():
#     print(f"{from_spk} → {to_spk} : {count} mal")

############## symmetrische Wechselzählung mit dict
# from collections import Counter
# import pandas as pd
#
# # Dein Dictionary mit Segmenten
# my_segs = spk_segs.iloc[0]
#
# # Alle Segmente sammeln
# entries = []
# for spk, segs in my_segs.items():
#     for start, end in segs:
#         entries.append({'speaker': spk, 'start': start, 'end': end})
#
# # Nach Startzeit sortieren
# entries.sort(key=lambda x: x['start'])
#
# # Zeitfenster in Sekunden
# time_window = 4.0
#
# # Symmetrische Zählung
# switch_counts = Counter()
#
# for i in range(len(entries) - 1):
#     current = entries[i]
#     next_seg = entries[i + 1]
#
#     gap = next_seg['start'] - current['end']
#     if 0 < gap <= time_window:
#         # Sprecherpaar ohne Richtung
#         pair = tuple(sorted([current['speaker'], next_seg['speaker']]))
#         switch_counts[pair] += 1
#
# # Alle Sprecher sammeln
# speakers = sorted(my_segs.keys())
#
# # Leere Matrix aufbauen
# matrix = pd.DataFrame(0, index=speakers, columns=speakers)
#
# # Zähler eintragen (symmetrisch in beide Richtungen)
# for (spk_a, spk_b), count in switch_counts.items():
#     matrix.loc[spk_a, spk_b] = count
#     matrix.loc[spk_b, spk_a] = count
#
# print(matrix)
#
# # Transpose and write  my_segs as rows
# with open("my_segs.csv", "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(my_segs.keys())  # header
#     writer.writerows(zip(*my_segs.values()))

merged_spk_segs_df = merge_consecutive_segments_per_speakers(spk_segs.iloc[0])
merged_spk_segs_df.to_csv("merged_segs.csv")
########################################################################### symmetrische Wechselzählung mit dataframe
from collections import Counter
import pandas as pd

# Beispiel: spk_segs ist ein DataFrame mit Spalten speaker, start, end
# spk_segs = pd.DataFrame([...])

# Alle Segmente sammeln und nach Startzeit sortieren
entries = merged_spk_segs_df.sort_values("start").reset_index(drop=True).to_dict(orient="records")

# Zeitfenster in Sekunden
time_window = 4.0

# Symmetrische Zählung
switch_counts = Counter()

for i in range(len(entries) - 1):
    current = entries[i]
    next_seg = entries[i + 1]

    gap = next_seg['start'] - current['end']
    if 0 < gap <= time_window:
        # Sprecherpaar ohne Richtung
        pair = tuple(sorted([current['speaker'], next_seg['speaker']]))
        switch_counts[pair] += 1

# Alle Sprecher sammeln
speakers = sorted(merged_spk_segs_df['speaker'].unique())

# Leere Matrix aufbauen
matrix = pd.DataFrame(0, index=speakers, columns=speakers)

# Zähler eintragen (symmetrisch)
for (spk_a, spk_b), count in switch_counts.items():
    matrix.loc[spk_a, spk_b] = count
    matrix.loc[spk_b, spk_a] = count

print(matrix)

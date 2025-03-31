import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

# 1. Simulierte Daten vorbereiten
# Zeitachse (in Sekunden)
time = np.linspace(0, 10, 1000)  # 10 Sekunden
# Simulierte Audiowellenform
audio_signal = np.sin(2 * np.pi * 5 * time) + 0.5 * np.random.randn(len(time))

# Simulierte Speaker Diarization (SD) Ergebnisse und Ground Truth
# Format: [(start, end, speaker_id), ...]
sd_results = [(0, 2, 1), (2.5, 3, 2), (3.5, 4, 1), (5, 7, 1), (8, 9, 2)]
ground_truth = [(0, 1.5, 1), (2, 3, 2), (4, 6, 1), (7, 8, 2), (9, 10, 3)]

# Farben für die Sprecher
colors = {1: 'red', 2: 'blue', 3: 'yellow'}

# 2. Layout erstellen
fig = plt.figure(figsize=(12, 8))

# 3. Obere Reihe: Szenenbilder (Platzhalter)
# Wir erstellen 4 Subplots für die Bilder
for i in range(4):
    ax = fig.add_subplot(5, 4, i + 1)
    # Platzhalterbild (du kannst hier echte Bilder einfügen)
    placeholder_img = np.ones((100, 100, 3)) * (i / 4)  # Simuliertes Bild
    ax.imshow(placeholder_img)
    ax.axis('off')
    ax.set_title(f"Person {i+1}")

# 4. Mittlere Reihe: Audiowellenform
ax_audio = fig.add_subplot(5, 1, 3)
ax_audio.plot(time, audio_signal, color='blue')
ax_audio.set_xlim(0, 10)
ax_audio.set_title("Audio Waveform")
ax_audio.set_xlabel("Time (s)")
ax_audio.set_ylabel("Amplitude")

# 5. Untere Reihen: SD Result und Ground Truth
# SD Result
ax_sd = fig.add_subplot(5, 1, 4)
ax_sd.set_xlim(0, 10)
ax_sd.set_ylim(0, 1)
ax_sd.set_title("SD Result")
ax_sd.set_xlabel("Time (s)")
ax_sd.set_yticks([])

for start, end, speaker in sd_results:
    ax_sd.add_patch(Rectangle((start, 0), end - start, 1, facecolor=colors[speaker]))

# Ground Truth
ax_gt = fig.add_subplot(5, 1, 5)
ax_gt.set_xlim(0, 10)
ax_gt.set_ylim(0, 1)
ax_gt.set_title("Ground Truth")
ax_gt.set_xlabel("Time (s)")
ax_gt.set_yticks([])

for start, end, speaker in ground_truth:
    ax_gt.add_patch(Rectangle((start, 0), end - start, 1, facecolor=colors[speaker]))

# 6. Layout anpassen
plt.tight_layout()
plt.show()
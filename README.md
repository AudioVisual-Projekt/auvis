
# Projektgruppe:  Audiovisuelle Gesprächsanalyse

Dieses Repository enthält das **AUVIS‑System**, das im Rahmen einer Projektarbeit im Sommer- und Wintersemester 25/26 an der Fachhochschule Südwestfalen entwickelt und als Beitrag zur  
[**CHiME‑9 Multi‑Modal Context‑aware Recognition (MCoRec) Challenge**](https://www.chimechallenge.org/current/task1/index) eingereicht wurde.

## 1. Aufgabenstellung

Die CHiME-9 MCoRec Challenge fordert robuste audiovisuelle Spracherkennungssysteme für unskriptierte „In-the-Wild"-Szenarien: 360°-Videoaufnahmen mit mehreren Sprechern, die sich frei im Raum bewegen. Ziel ist die korrekte Transkription und gleichzeitige Zuordnung der Sprache zu Konversationsgruppen. Die Bewertung erfolgt über die **Joint ASR-Clustering Error Rate (JACR)**, die Transkriptionsgenauigkeit und Sprecherzuordnung gemeinsam bewertet.

## 2. Systemübersicht

Das AUVIS-System basiert auf der [offiziellen CHiME‑9‑Baseline](https://github.com/MCoRec/mcorec_baseline), nutzt die [bereitgestellten Challenge-Daten](https://huggingface.co/datasets/MCoRecChallenge/MCoRec) und folgt einer modularen, sequenziellen Pipeline:

**Eingabe:** 360° Video + Audio

- **Subtask 1:** Active Speaker Detection (ASD)
- **Subtask 2:** Visuelles Tracking & Mouth Cropping
- **Subtask 3:** Zeitliche Segmentierung & Chunking
- **Subtask 4:** Audio‑Visuelle Spracherkennung (AV‑HuBERT)
- **Subtask 5:** Konversations‑Clustering

**Ausgabe:** Sprecherzugeordnete Transkripte und Konversations‑IDs

## 3. Zentrale Beiträge dieser Arbeit

Für jeden der fünf Subtasks der Pipeline wurden systematisch Experimente durchgeführt, um Verbesserungen gegenüber der Baseline zu erzielen.

- Entkopplung der Segmentierungsparameter zwischen **AVSR‑ und Clustering‑Pipeline**
- Gemeinsame Optimierung von:
  - ASD‑Segmentierungsparametern
  - Clustering‑Parametern
- Systematische Evaluation und bewusste Verwerfung von:
  - alternativen ASD‑Architekturen
  - MediaPipe‑basierter Landmark‑Extraktion
  - LLM‑gestützter Transkript‑Nachkorrektur
  - alternatives ASR‑Modell (Whisper‑Flamingo)
- Nachweis, dass **System‑ und Schnittstellenoptimierung** größere Leistungsgewinne erzielt
  als der Einsatz komplexerer Einzelmodelle


## 4. Ergebnisse (Development‑Datensatz)

| Metrik | Baseline | AUVIS (optimiert) |
|------|---------|------------------|
| Clustering‑F1 (pro Sprecher) | 0,789 | **0,906** |
| Relative Verbesserung Clustering | – | **+14,8 %** |
| Speaker WER | 0,4987 | **0,4943** |
| Relative Reduktion WER | – | **−0,88 %** |
| Joint ASR‑Clustering Error Rate | 0,355 | **0,294** |
| Relative Reduktion JACR | – | **−17,09 %** |

Die signifikanten Leistungsgewinne resultieren primär aus der **Optimierung des Konversations‑Clusterings**
und der vorgelagerten ASD‑Segmentierung.

## 4. Repository-Struktur
```
auvis_system/          # Modifizierte CHiME-9-Baseline (Kernsystem)
├── script/            # angepasste Inference, Training und Evaluationsskripte
├── src/               # Modellarchitekturen und Hilfsfunktionen
├── subtask_1_additional_files/  # Experimente Subtask 1
├── subtask_2/         # Experimente Subtask 2
├── subtask_3_experiments/       # Experimente Subtask 3
├── subtask_4_experiments/       # Experimente Subtask 4
└── subtask_5_experiments/       # Experimente Subtask 5
```

## 5. Projektsetup 

### 1. Virtuelle Umgebung im Ordner ".venv" erstellen
python -m venv .venv

### 2. Virtuelle Umgebung aktivieren
linux/macOS:
source .venv/bin/activate
Windows  - CMD
.venv\Scripts\activate

### 3. Pakete aus requirements.txt installieren
pip install -r requirements.txt

### 4. Für die Audio Diarization wird ein Huggingface Token benötigt
in die ".env" - Datei eintragen

### 5. ffmpeg muss auf dem System installiert werden
https://www.ffmpeg.org/

Siehe auch die [Setup-Dokumentation der CHiME-9-Challenge](https://github.com/MCoRec/mcorec_baseline).

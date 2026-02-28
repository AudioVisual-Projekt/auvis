
# Projektgruppe:  Audiovisuelle Gesprächsanalyse

Dieses Repository enthält das **AUVIS‑System**, das im Rahmen einer Projektarbeit im Sommer- und Wintersemester 25/26 an der Fachhochschule Südwestfalen entwickelt und als Beitrag zur  
[**CHiME‑9 Multi‑Modal Context‑aware Recognition (MCoRec) Challenge**](https://www.chimechallenge.org/current/task1/index) eingereicht wurde.

Der Fokus des Systems liegt auf der robusten **audiovisuellen Spracherkennung** und dem **Konversations‑Clustering**
in unstrukturierten, realitätsnahen Mehrsprecher‑Szenarien („in‑the‑wild“) auf Basis von 360°‑Video‑ und Audiodaten.


## 1. Systemübersicht

Das AUVIS‑System basiert auf der offiziellen CHiME‑9‑Baseline und folgt einer modularen, sequenziellen Pipeline:

**Eingabe:** 360° Video + Audio

- **Subtask 1:** Active Speaker Detection (ASD)
- **Subtask 2:** Visuelles Tracking & Mouth Cropping
- **Subtask 3:** Zeitliche Segmentierung & Chunking
- **Subtask 4:** Audio‑Visuelle Spracherkennung (AV‑HuBERT)
- **Subtask 5:** Konversations‑Clustering

**Ausgabe:** Sprecherzugeordnete Transkripte und Konversations‑IDs

Im Gegensatz zu vielen alternativen Ansätzen liegt der Schwerpunkt **nicht** auf dem Austausch einzelner Modelle,
sondern auf der **Optimierung der Schnittstellen und Parameterkopplungen** zwischen den Modulen.


## 2. Zentrale Beiträge dieser Arbeit

- Entkopplung der Segmentierungsparameter zwischen **ASR‑ und Clustering‑Pipeline**
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


## 3. Ergebnisse (Development‑Datensatz)

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

## 4. Ways of working
Ways of working ist im Wiki niedergeschrieben.
[klick hier](https://github.com/AudioVisual-Projekt/auvis/wiki/Ways-of-working)


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

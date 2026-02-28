
# Projektgruppe:  Audiovisuelle Gesprächsanalyse  SoSe25+WiSe25/26

Dieses Repository enthält das **AUVIS‑System**, das im Rahmen einer Masterarbeit entwickelt und als Beitrag zur  
**CHiME‑9 Multi‑Modal Context‑aware Recognition (MCoRec) Challenge** eingereicht wurde.

Der Fokus des Systems liegt auf der robusten **audiovisuellen Spracherkennung** und dem **Konversations‑Clustering**
in unstrukturierten, realitätsnahen Mehrsprecher‑Szenarien („in‑the‑wild“) auf Basis von 360°‑Video‑ und Audiodaten.



Ways of working ist im Wiki niedergeschrieben.
[klick hier](https://github.com/AudioVisual-Projekt/auvis/wiki/Ways-of-working)



## Setup the project

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

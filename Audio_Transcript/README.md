# Audio_Transcript mit Whisper für die Textextraktion und Pyannote für die Sprecherzuordnung
D. Schönenberg

Der Code wurde aus dem Tutorial von C. Groll übernommen 
https://www.youtube.com/watch?v=aiFUJU-dXhI&list=PLGkspzSREmDeyY4rpkab3SLiSBUH-c0iD

Nach Installation der Packages in eine virtuelle Umgebung .venv

pip install --editable .

Dann kann das Python Script im VS Code Terminal aufgerufen werden mit 

from_wav

oder

from_url 

bzw. in der Windows cmd wie folgt aufgerufen werden:

[C:\Users\...\]Audio_Transcript\.venv\Scripts\from_wav

[C:\Users\...\]Audio_Transcript\.venv\Scripts\from_url

es folgt jeweils (etwas verzögert) eine Eingabeaufforderung für den Audio-Input (wav-Datei oder url)

danach folgt eine zweite Eingabeaufforderung, welches Whisper Modell verwendet werden soll, 'base' ist default, das Ergebnis ist bei größeren Modellen genauer.


Es wird ein Huggingface Token benötigt!
Dieser wird in der Datei .env abgelegt, die hier beigefügt ist, jedoch ohne Token.


Die Main-Datei ist click_app.py

Funktionen sind definiert in src/audio_utils.py 

sowie in /models/whisper.py, /models/pyannote.py und /models/conversation_transcription.py




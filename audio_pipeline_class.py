######################################
# aktuelles Arbeitsverzeichnis ausgeben und das Elternverzeichnis dieser Datei  
# in die Python - Suchpfade (sys.path) aufnehmen
# => Module / Pakete aus dem Elternverzeichnis können so gefunden und importiert werden
#
import os
import sys
# print("Current working directory:", os.getcwd())
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
##################################################################################
import subprocess
import soundfile as sf
from audio_diarization_class import Audio_Diarization


class Audio_Pipeline(object):
    def __init__(self, input_fname):
        self.temp_files = []
        self.input_fname = input_fname
        self.input_fpath = os.path.abspath(self.input_fname)
        self.audio_only_fpath = self.get_audio_path() if self.has_videostream() else self.input_fpath
        self.wav_fpath = self.get_wav_path()
        self.diarization = self.audio_pipeline()
        
    def has_videostream(self):
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            self.input_fpath
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return bool(result.stdout.strip())
        
    def extract_audio(self):
        try:
            cmd = [
                "ffmpeg",
                "-y",              # überschreibt ohne Rückfrage !!!
                "-i", self.input_fpath,
                "-vn",
                "-acodec", "copy",
                self.audio_only_fpath
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=30         # Timeout gesetzt
            )
            if result.returncode != 0:
                print("Fehler bei ffmpeg:", result.stderr)
            
            self.register_temp_file(self.audio_only_fpath)
        except subprocess.TimeoutExpired:
            print("Timeout: ffmpeg hat zu lange gebraucht.")
        except Exception as e:
            print("Fehler bei ffmpeg:", str(e))
            
    def get_audio_path(self):
        input_dir = os.path.dirname(self.input_fpath)
        fname_ohne_ext, ext = os.path.splitext(os.path.basename(self.input_fname)) 
        audio_fpath = os.path.join(input_dir, f"{fname_ohne_ext}_audio_only{ext}")
        return audio_fpath

    def get_wav_path(self):
        audio_dir = os.path.dirname(self.audio_only_fpath)
        fname_ohne_ext, ext = os.path.splitext(os.path.basename(self.audio_only_fpath)) 
        wav_fpath = os.path.join(audio_dir, f"{fname_ohne_ext}.wav")
        return wav_fpath

    def convert_audio_to_wav(self):     ## prepare audio for pyannote
        """
        Konvertiert eine Datei in 16kHz, mono WAV für pyannote.audio, wenn nötig.
        Gibt Pfad zur WAV-Datei zurück.
        """
        try:
            # Prüfen ob Datei bereits ein passendes WAV ist
            if self.input_fpath.lower().endswith(".wav"):
                with sf.SoundFile(self.input_fpath) as f:
                    if f.channels == 1 and f.samplerate == 16000:
                        print("Datei ist bereits mono WAV mit 16kHz.")
                        return self.input_fpath  # nichts zu tun

            # Konvertierung mit ffmpeg
            print(f"Konvertiere {self.input_fpath} → {self.wav_fpath} ...")
            cmd = [
                "ffmpeg",
                "-y",                # überschreiben
                "-i", self.input_fpath,
                "-ac", "1",          # mono
                "-ar", "16000",      # 16kHz
                self.wav_fpath
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            self.register_temp_file(self.wav_fpath)
            return self.wav_fpath

        except Exception as e:
            print("Fehler bei der Vorbereitung:", str(e))
            return None



    def audio_pipeline(self):  
        ## extract Audio if Input file has Video and Audio combi  
        if self.has_videostream():
            self.extract_audio()
        ## Convert Audio to wav format
        self.convert_audio_to_wav()
        ## Audio_diarization
        self.diarization = Audio_Diarization(self.wav_fpath)
        transcript = self.diarization.wav_to_transcript()
        self.cleanup_temp_files()
        return self.diarization


    def register_temp_file(self, path):
        """Speichert Pfad zur späteren Löschung."""
        self.temp_files.append(path)

    def cleanup_temp_files(self):
        """Löscht alle registrierten temporären Dateien."""
        for path in self.temp_files:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"Temp-Datei gelöscht: {path}")
            except Exception as e:
                print(f"Fehler beim Löschen von {path}: {e}")
        self.temp_files.clear()

    def __del__(self):
        """Optional: automatische Bereinigung beim Löschen der Instanz."""
        self.cleanup_temp_files()

if __name__ == "__main__":    
    
    input_fname = "WDR_video_short.mp4"
    audio_pipe = Audio_Pipeline(input_fname)
    ## Diarization Ergebnisse  unter
    ## audio_pipe.diarization.text_speaker_df
    ## audio_pipe.diarization.output_str  
    ## etc. (siehe Audio_Diarization_Class)

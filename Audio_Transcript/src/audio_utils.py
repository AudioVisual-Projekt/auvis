import pyaudio
import wave
import requests
import tempfile
from pydub import AudioSegment

def record_to_wav(RECORD_SECONDS, WAVE_OUTPUT_FILENAME):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("recording")

    frames = []

    for i in range(0, int(RATE/CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
        
    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def download_mp3(audio_url, fname=None):
    # download
    doc = requests.get(audio_url)
    
    if fname:
        this_temp_file_name = fname
        with open(fname,'wb') as f:
            f.write(doc.content)
            
    else:
        # create temp file
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        this_temp_file_name = temp_file.name
        
        #write to file
        temp_file.write(doc.content)
    
    return this_temp_file_name

def transform_mp3_to_wav(mp3_fname, output_fname=None):
    sound = AudioSegment.from_mp3(mp3_fname) # load source
    
    if output_fname:
        this_temp_file_name = output_fname
        sound.export(output_fname, format="wav")
            
    else:
        # create temp file
        temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        this_temp_file_name = temp_file.name
        sound.export(this_temp_file_name, format="wav")  
    
    return this_temp_file_name


def crop_wav(fname, output_fname, start_frame=0, n_frames=60000):
    
    sound = AudioSegment.from_wav(fname)
    
    sound = sound.set_channels(1) # mono
    sound = sound.set_frame_rate(16000) # 16000Hz

    # Extract the n_frames from start_frame (60000 = 60 seconds)
    excerpt = sound[start_frame:(start_frame + n_frames)]
    
    # write to disc
    excerpt.export(output_fname, format="wav")


if __name__ == '__main__':
    RECORD_SECONDS = 5
    WAVE_OUTPUT_FILENAME = "voice_2.wav"
    record_to_wav(RECORD_SECONDS, WAVE_OUTPUT_FILENAME)
    
    # download podcast and save it to mp3 file

    audio_url = "https://dts.podtrac.com/redirect.mp3/tracking.swap.fm/track/0bDcdoop59bdTYSfajQW/pdst.fm/e/stitcher.simplecastaudio.com/2be48404-a43c-4fa8-a32c-760a3216272e/episodes/37f33f60-213b-453e-b7b1-f516284376da/audio/128/default.mp3?aid=rss_feed&amp;awCollectionId=2be48404-a43c-4fa8-a32c-760a3216272e&amp;awEpisodeId=37f33f60-213b-453e-b7b1-f516284376da&amp;feed=Y8lFbOT4"
    tmp_fname = download_mp3(audio_url, fname='freakonomics.mp3')

    mp3_fname = 'freakonomics.mp3'
    wav_fname = 'freakonomics.wav'
    transform_mp3_to_wav(mp3_fname, output_fname=wav_fname)
    
    source = wav_fname
    crop_wav(source, 'freakonomics_short.wav', start_frame=0, n_frames=240000)
    
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

# ===========================================================
# 📈 WAVEFORM PLOT INSPECTION GUIDE
# ===========================================================
# Aspect            | What it measures                                   | Where to see it       | Interpretation and Action
# -----------------------------------------------------------------------------------------------------
# Sample Rate       | Number of samples per second                       | Printed info          | If ≠16kHz → Resample to 16kHz
# Channels          | Mono vs Stereo (1s vs 2 channels)                   | Printed info          | Stereo → Convert to mono (average channels)
# Duration          | Total length of audio in seconds                   | Printed info          | Very long (>10 min) → Consider splitting for efficiency
# Loudness Range    | Peak amplitude values [-1, +1]                     | Printed info          | Peak > 1 or = ±1 → Clipping → Apply dynamic range compression or re-record
#                   |                                                     |                       | Peak ≪ 1 (e.g., max 0.3) → Normalize audio
# Flat Segments     | Long regions of no activity                        | Waveform view         | Long silences (>5s) at start/end → Trim
# Speech Bursts     | Active parts with clear ups/downs                  | Waveform view         | Natural speech should have visible dynamics; if flat, check recording
# Clipping Spikes   | Sharp, square peaks at max/min                     | Waveform view         | If visible → Possible audio damage → Consider re-record or strong smoothing
# Channel Imbalance | Big difference between channels                   | Stereo waveform view  | If one channel is much louder → Prefer the louder one or average carefully
# ===========================================================

# ===========================================================
# 🎛️ SPECTROGRAM PLOT INSPECTION GUIDE
# ===========================================================
# Aspect            | What it measures                                   | Where to see it       | Interpretation and Action
# -----------------------------------------------------------------------------------------------------
# Speech Energy Range | Frequency range where speech energy is present  | Spectrogram vertical  | Normal speech: 100 Hz – 6000 Hz. No energy there → Check recording quality.
# Noise Floor       | Minimum background energy level                   | Color scale (dB)      | If background ≥-50 dB → Noisy → Apply stronger denoising
# Silence Gaps      | Very dark areas between speech parts               | Spectrogram view      | Clean speech should show clear gaps. If no gaps → Consider noise suppression.
# Background Hum    | Constant horizontal lines at low freq (50/60 Hz)   | Spectrogram view      | If visible → Apply notch filter or denoiser
# Reverberation     | Smearing upwards in time after speech bursts       | Spectrogram view      | If strong → Use dereverberation model if needed
# High-Frequency Noise | Random speckles >10kHz                         | Top of spectrogram    | If strong → Low-pass filter or denoise
# Clicks/Pops       | Very narrow vertical lines across all frequencies  | Spectrogram view      | If visible → Click/pop removal needed
# Dynamic Range     | Bright vs dark contrast                            | Overall spectrogram   | Low contrast → Might need amplification before model input
# ===========================================================

# Load the audio file
file_path = "auvis\src\backend\services\muffled-talking-6161.mp3"
def inspect_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None, mono=False)
    duration = librosa.get_duration(y=audio, sr=sr)
    channels = 1 if audio.ndim == 1 else audio.shape[0]

    print(f"Filename: {file_path}")
    print(f"Sample Rate: {sr} Hz")
    print(f"Channels: {channels}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Amplitude range: min={audio.min():.2f}, max={audio.max():.2f}")

    # Plot waveform
    plt.figure(figsize=(10, 4))
    if channels == 1:
        plt.plot(audio)
    else:
        plt.plot(audio[0], alpha=0.6, label="Channel 1")
        plt.plot(audio[1], alpha=0.6, label="Channel 2")
        plt.legend()
    plt.title("Waveform")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()


    # Before creating spectrogram, check if stereo and convert
    if audio.ndim > 1:
        audio = np.mean(audio, axis=0)


    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()

inspect_audio(file_path)

import torchaudio

def convert_to_16khz_mono(input_path, output_path, target_sr=16000):
    # Load audio
    waveform, sample_rate = torchaudio.load(input_path)

    print(f"Original sample rate: {sample_rate} Hz")
    print(f"Original number of channels: {waveform.shape[0]}")

    # Convert stereo to mono if needed
    if waveform.shape[0] > 1:
        print("Converting to mono...")
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sample_rate != target_sr:
        print(f"Resampling from {sample_rate} Hz to {target_sr} Hz...")
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)

    # Fix shape if needed (unsqueeze)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)

    # Save the converted file
    torchaudio.save(output_path, waveform, target_sr)
    print(f"Saved converted file to {output_path}")

    # Proof: Reload and print confirmation
    new_waveform, new_sr = torchaudio.load(output_path)
    print(f"✅ Confirmed sample rate: {new_sr} Hz")
    print(f"✅ Confirmed number of channels: {new_waveform.shape[0]}")

convert_to_16khz_mono("/content/muffled-talking-6161.mp3", "/content/harvard_16khz_mono.wav")

import soundfile as sf

def inspect_and_fix_loudness(input_path, output_path, normalize_threshold=0.9):
    audio, sr = librosa.load(input_path, sr=None)

    peak = np.max(np.abs(audio))
    print(f"Original peak amplitude: {peak:.3f}")

    if peak >= 1.0:
        print("⚠️ Warning: Audio is clipped or very close to clipping. Skipping normalization.")
        sf.write(output_path, audio, sr)
        print(f"Saved original audio without changes to {output_path}")

    elif peak < normalize_threshold:
        print(f"🔹 Audio is too quiet (peak {peak:.3f}). Applying peak normalization...")
        audio = audio / peak  # Scale up so that max amplitude becomes 1.0
        sf.write(output_path, audio, sr)
        print(f"✅ Saved normalized audio to {output_path}")

    elif peak >= normalize_threshold:
        print(f"🔹 Audio has good peak ({peak:.3f}) but might benefit from Dynamic Range Compression...")
        audio = dynamic_range_compression(audio)
        sf.write(output_path, audio, sr)
        print(f"✅ Saved compressed audio to {output_path}")

def dynamic_range_compression(audio, threshold=0.5, ratio=4.0):
    """
    Apply simple dynamic range compression.
    Threshold: amplitude above which compression starts.
    Ratio: how much to reduce dynamics above threshold.
    """
    compressed = np.copy(audio)
    mask = np.abs(compressed) > threshold
    compressed[mask] = np.sign(compressed[mask]) * (threshold + (np.abs(compressed[mask]) - threshold) / ratio)
    return compressed

inspect_and_fix_loudness("/content/harvard_16khz_mono.wav", "/content/harvard_loudness_fixed.wav")

!pip install noisereduce

import noisereduce as nr

def apply_denoising_if_needed(audio, sr, threshold_db=-50):
    """Apply denoising only if the noise floor is too high"""
    S = librosa.stft(audio)
    db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    noise_floor = np.median(db)

    if noise_floor > threshold_db:
        print(f"🔹 Noise floor is {noise_floor:.2f} dB > {threshold_db} dB — denoising applied.")
        return nr.reduce_noise(y=audio, sr=sr)
    else:
        print(f"✅ Noise floor is acceptable ({noise_floor:.2f} dB) — skipping denoising.")
        return audio

# Load your already normalized audio
audio, sr = librosa.load("/content/harvard_loudness_fixed.wav", sr=None)

# Apply conditional denoising
denoised_audio = apply_denoising_if_needed(audio, sr, threshold_db=-50)

# Save result
sf.write("/content/harvard_denoised.wav", denoised_audio, sr)
print("✅ Final denoised audio saved.")

def plot_waveforms(original_audio, fixed_audio, sr, title_original="Original", title_fixed="Processed"):
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Original waveform
    axs[0].plot(np.linspace(0, len(original_audio) / sr, num=len(original_audio)), original_audio)
    axs[0].set_title(title_original)
    axs[0].set_ylabel("Amplitude")
    axs[0].grid(True)

    # Processed waveform
    axs[1].plot(np.linspace(0, len(fixed_audio) / sr, num=len(fixed_audio)), fixed_audio, color="orange")
    axs[1].set_title(title_fixed)
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Amplitude")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()

# Load original and fixed audio
orig_audio, sr = librosa.load("/content/muffled-talking-6161.mp3", sr=None)
fixed_audio, _ = librosa.load("/content/harvard_loudness_fixed.wav", sr=None)

# Plot them
plot_waveforms(orig_audio, fixed_audio, sr, title_original="Original Audio", title_fixed="Normalized/Compressed Audio")
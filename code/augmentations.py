import os
import random
import librosa
import soundfile as sf
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import scipy.signal as signal

# Define paths
dataset_path = r"C:\Users\USER\Documents\My Projects\Main Project\app\uploads\Original Balanced  RAVDESS"  # Path to the RAVDESS dataset
output_path = r"C:\Users\USER\Documents\My Projects\Main Project\app\uploads\Augmented Balanced RAVDESS 2"  # Path to save augmented dataset
os.makedirs(output_path, exist_ok=True)

def add_reverb(audio, sr, decay=0.3):
    impulse = np.zeros(int(sr * 0.5))  # 0.5 seconds
    impulse[0] = 1  # Dirac delta
    impulse[int(sr * decay)] = 0.5  # Simulated echo
    reverb_audio = signal.convolve(audio, impulse, mode='full')[:len(audio)]
    return reverb_audio

def apply_eq(audio, sr, lowcut=300.0, highcut=3000.0):
    sos = signal.butter(10, [lowcut, highcut], btype='band', fs=sr, output='sos')
    filtered_audio = signal.sosfilt(sos, audio)
    return filtered_audio

def clip_audio(audio, clip_threshold=0.3):
    clipped_audio = np.clip(audio, -clip_threshold, clip_threshold)
    return clipped_audio

def add_echo(audio, sr, delay=0.2, decay=0.5):
    delay_samples = int(sr * delay)
    echo_audio = np.pad(audio, (delay_samples, 0)) * decay
    return audio + echo_audio[:len(audio)]

def volume_scaling(audio, scale_factor):
    return audio * scale_factor

def random_crop(audio, crop_fraction=0.1):
    crop_size = int(len(audio) * crop_fraction)
    start = np.random.randint(0, len(audio) - crop_size)
    cropped_audio = audio[start:start + crop_size]
    # Extend cropped audio to match original length
    cropped_audio = np.pad(cropped_audio, (0, max(0, len(audio) - len(cropped_audio))))
    return cropped_audio

def augment_audio(file_path, sample_rate=16000):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    augmented_audios = [(audio, 'original')]

    # Noise Addition
    for noise_level in [0.02, 0.04]:
        noise = np.random.normal(0, noise_level, audio.shape)
        augmented_audios.append((audio + noise, f'noise-added-{noise_level}'))

    # Time Stretching
    for rate in [0.9, 1.1]:
        time_stretched = librosa.effects.time_stretch(audio, rate=rate)
        augmented_audios.append((time_stretched[:len(audio)], f'time-stretched-{rate}'))

    # Pitch Shifting
    for n_steps in [2, -2, 4, -4]:
        pitch_shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
        augmented_audios.append((pitch_shifted, f'pitch-shifted-{n_steps}'))

    # Dynamic Range Compression
    pre_emphasized = librosa.effects.preemphasis(audio)
    augmented_audios.append((pre_emphasized, 'pre-emphasized'))

    # SpecAugment
    spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    
    for _ in range(2):  # Frequency Masking
        freq_mask = np.copy(spec)
        num_mel_bands = freq_mask.shape[0]
        f = random.randint(0, num_mel_bands - 1)
        width = random.randint(1, num_mel_bands // 4)
        freq_mask[f:f+width, :] = 0
        freq_mask_audio = librosa.feature.inverse.mel_to_audio(freq_mask, sr=sr)
        augmented_audios.append((freq_mask_audio[:len(audio)], f'freq-masked-band-{f}-width-{width}'))

    for _ in range(2):  # Time Masking
        time_mask = np.copy(spec)
        num_time_steps = time_mask.shape[1]
        t = random.randint(0, num_time_steps - 1)
        width = random.randint(1, num_time_steps // 8)
        time_mask[:, t:t+width] = 0
        time_mask_audio = librosa.feature.inverse.mel_to_audio(time_mask, sr=sr)
        augmented_audios.append((time_mask_audio[:len(audio)], f'time-masked-start-{t}-width-{width}'))


    
    # Reverberation
    reverb_audio = add_reverb(audio, sr)
    augmented_audios.append((reverb_audio, 'reverberated'))

    # Equalization
    eq_audio = apply_eq(audio, sr)
    augmented_audios.append((eq_audio, 'equalized'))

    # Clipping
    clipped_audio = clip_audio(audio)
    augmented_audios.append((clipped_audio, 'clipped'))

    # Echo
    echo_audio = add_echo(audio, sr)
    augmented_audios.append((echo_audio, 'echo-added'))

    # Volume Scaling
    for scale in [0.5, 1.5]:
        scaled_audio = volume_scaling(audio, scale)
        augmented_audios.append((scaled_audio, f'volume-scaled-{scale}'))

    # Random Cropping
    cropped_audio = random_crop(audio)
    augmented_audios.append((cropped_audio, 'random-cropped'))

    return augmented_audios, sr

# Function to process a single file
def process_file(file_path, output_dir):
    try:
        augmented_audios, sr = augment_audio(file_path)
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        for audio, aug_label in augmented_audios:
            save_path = os.path.join(output_dir, f"{base_name}-{aug_label}.wav")
            sf.write(save_path, audio, sr)
            print(f"Saved: {save_path}")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# Walk through dataset and apply augmentations
with ThreadPoolExecutor() as executor:
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, dataset_path)
                output_dir = os.path.join(output_path, relative_path)
                executor.submit(process_file, file_path, output_dir)

print(f"Augmented dataset saved at: {output_path}")
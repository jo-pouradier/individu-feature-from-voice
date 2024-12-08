import librosa
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm

def load_wav(file_path: str):
    """
    Load a wav file using librosa
    :param file_path: Path to the wav file
    :return: Tuple of the audio data and the sample rate
    """
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    return audio_data, sample_rate


def get_spectrogram(audio):
    # Short-time Fourier transform (STFT).
    stft = abs(librosa.stft(audio))

    # Convert an amplitude spectrogram to dB-scaled spectrogram.
    spectrogram = librosa.amplitude_to_db(stft)

    return spectrogram


def get_mfcc(audio, sampling_rate, n_mfcc=20):
    mfcc = librosa.feature.mfcc(y=audio, sr=sampling_rate, n_mfcc=n_mfcc)
    return mfcc


def clip_audio(audio, duration=10, repeat=False):
    """
    Clip the audio to a specific duration
    :param audio: Audio data
    :param duration: Duration to clip the audio to
    :return: Clipped audio data
    """
    # Get the number of samples for the desired duration
    n_samples = int(duration * sample_rate)
    if len(audio) < n_samples:
        # Repeat the audio if it is shorter than the desired duration
        if repeat:
            n_repeats = n_samples // len(audio)
            n_samples = n_repeats * len(audio)
            audio = np.tile(audio, n_repeats)
        else:
            # Pad the audio with zeros if it is shorter than the desired duration
            n_zeros = n_samples - len(audio)
            audio = np.pad(audio, (0, n_zeros))
    # Clip the audio
    clipped_audio = audio[:n_samples]
    return clipped_audio


def extract_features(file_name, base_path, output_path, save=False):
    audio_data, sample_rate = load_wav(os.path.join(base_path, file_name))
    spectrogram = get_mfcc(audio_data, sample_rate, n_mfcc=40)
    features = []
    for el in spectrogram:
        features.append(np.mean(el))
    features = np.array(features)
    if save:
        np.save(os.path.join(output_path, file_name.split(".")[0]), features)
    return file_name, features


if __name__ == "__main__":
    # apply_async example
    base_path = r"/mnt/d/datasets/cv-corpus-19.0-2024-09-13-fr/cv-corpus-19.0-2024-09-13/fr/clips/"
    start_time = time.time()
    num_processes = 8
    data = pd.read_csv(os.path.join("../","validated_filtered_500_per_age.csv"), sep=',')
    paths = data['path'].to_list()
    # paths = paths[:8]
    print(f"Number of paths: {len(paths)}")
    with Pool(num_processes) as p:
        async_result = p.starmap_async(extract_features, [(path, base_path, "processed") for path in paths])
        print('Waiting for results...')
        results = async_result.get()
    print(f"Multi --- {time.time() - start_time} seconds ---")
    df = pd.DataFrame(results, columns=["path", "mfcc_features"])
    # merge the features with the data
    data = data.merge(df, left_on='path', right_on='path')
    data.to_csv("features.csv", index=False)

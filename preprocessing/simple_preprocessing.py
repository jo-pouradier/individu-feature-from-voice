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

def extract_features(*args):
    file_name, base_path, output_path = args[0]
    audio_data, sample_rate = load_wav(os.path.join(base_path, file_name))
    mfcc = get_mfcc(audio_data, sample_rate, n_mfcc=40)
    features = []
    for el in mfcc:
        features.append(np.mean(el))
    features = np.array(features)
    return file_name, *features



if __name__ == "__main__":
    # apply_async example
    base_path = r"/mnt/d/datasets/cv-corpus-19.0-2024-09-13-fr/cv-corpus-19.0-2024-09-13/fr/clips/"
    start_time = time.time()
    num_processes = 8
    data = pd.read_csv(os.path.join("../","validated_filtered_5000_per_age.csv"), sep=',')
    paths = data['path'].to_list()
    # paths = paths[:1000]
    print(f"Number of paths: {len(paths)}")
    results = []
    with Pool(num_processes) as p:
        async_result =  tqdm(p.imap_unordered(extract_features, [(path, base_path, "processed") for path in paths]), total=len(paths))
        for res in async_result:
            results.append(res)
    print(f"Multi --- {time.time() - start_time} seconds ---")

    df = pd.DataFrame(results, columns=["path", *[f"mfcc_{i}" for i in range(40)]], dtype=object)
    # merge the features with the data
    data = data.merge(df, left_on='path', right_on='path')
    data.to_csv("features.csv", index=False)
    # plt.figure()
    # for i, row in data.iterrows():
    #     plt.plot(row['mfcc_features'])
    # plt.show()

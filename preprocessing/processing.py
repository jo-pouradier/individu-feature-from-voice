import os
import librosa
import numpy as np
from pydub import AudioSegment
import scipy.signal as signal
import matplotlib.pyplot as plt


# Preprocessing of audio files

class Processing:
    def __init__(self):
        pass

    def get_audio_files(self, path, extension:str = ".mp3"):
        """
        Get all the audio files in the given path
        :param path: Path to the folder containing the audio files
        :return: List of audio files
        """
        audio_files = []
        for file in os.listdir(path):
            if file.endswith(extension):
                audio_files.append(file)
        return audio_files



    def get_audio_data(self, path):
        """
        Get the audio data from the audio file
        :param path: Path to the audio file
        :return: Audio data
        """
        audio_data, sample_rate = librosa.load(path, sr=None)
        return audio_data, sample_rate


    def passebas_filter(self, audio_data, sample_rate, cutoff_frequency:int = 2000):
        """
        Apply low pass filter to the audio data
        :param audio_data: Audio data
        :param sample_rate: Sample rate of the audio data
        :param cutoff_frequency: Cutoff frequency for the low pass filter
        :return: Filtered audio data
        """
        nyquist_frequency = sample_rate / 2
        cutoff = cutoff_frequency / nyquist_frequency
        b, a = signal.butter(5, cutoff, btype='low')
        filtered_audio_data = signal.lfilter(b, a, audio_data)
        return filtered_audio_data
    

    def passehaut_filter(self, audio_data, sample_rate, cutoff_frequency:int = 2000):
        """
        Apply high pass filter to the audio data
        :param audio_data: Audio data
        :param sample_rate: Sample rate of the audio data
        :param cutoff_frequency: Cutoff frequency for the high pass filter
        :return: Filtered audio data
        """
        nyquist_frequency = sample_rate / 2
        cutoff = cutoff_frequency / nyquist_frequency
        b, a = signal.butter(5, cutoff, btype='high')
        filtered_audio_data = signal.lfilter(b, a, audio_data)
        return filtered_audio_data


    def mel_spectrogram(self, audio_data, sample_rate, n_mels:int = 128):
        """
        Compute the mel spectrogram of the audio data
        :param audio_data: Audio data
        :param sample_rate: Sample rate of the audio data
        :param n_mels: Number of mel bands
        :return: Mel spectrogram
        """
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sample_rate, n_mels=n_mels)
        return mel_spectrogram


    def mfcc(self, audio_data, sample_rate, n_mfcc:int = 13):
        """
        Compute the MFCC of the audio data
        :param audio_data: Audio data
        :param sample_rate: Sample rate of the audio data
        :param n_mfcc: Number of MFCC coefficients
        :return: MFCC
        """
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc)
        return mfcc
    

    def spectrogram_plot(self, audio_datas: list, sample_rate: list):
        """
        Display the spectrogram of the audio data
        :param audio_data: Audio data
        :param sample_rate: Sample rate of the audio data
        """
        # Créer une figure
        fig, axes = plt.subplots(len(audio_datas), 1, figsize=(10, 4 * len(audio_datas)))

        for audio_data, sr, i in zip(audio_datas, sample_rate, range(len(audio_datas))):

            S_dB = librosa.power_to_db(audio_data, ref=np.max)

            # Afficher le spectrogramme dans le sous-graphe correspondant
            ax = axes[i] if len(audio_datas) > 1 else axes  # Gère le cas d'un seul fichier
            img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', 
                                        fmax=sr / 2, ax=ax, cmap='coolwarm')
            ax.label_outer()  # Supprimer les étiquettes internes pour un affichage propre

        # Ajouter une barre de couleur commune
        # fig.colorbar(img, ax=axes, format='%+2.0f dB', location='right')
        return fig


    def spectral_substraction(self, audio_data, sample_rate:int = 3200, n_fft:int = 2048, hop_length:int = 512, power:float = 1.0, margin:float = 1.0):
        """
        Apply spectral substraction to the audio data
        :param audio_data: Audio data
        :param sample_rate: Sample rate of the audio data
        :param n_fft: Number of FFT components
        :param hop_length: Hop length
        :param power: Power coefficient
        :param margin: Margin coefficient
        :return: Filtered audio data
        """
        stft = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
        stft = np.abs(stft) ** power
        stft -= margin * np.mean(stft, axis=1, keepdims=True)
        stft[stft < 0] = 0
        filtered_audio_data = librosa.istft(stft, hop_length=hop_length)
        return filtered_audio_data
    

class PreprocessingPydub:

    def get_audio_data_pydub(self, path):
        """
        Get the audio data from the audio file using pydub
        :param path: Path to the audio file
        :return: Audio data
        """
        audio = AudioSegment.from_file(path)
        audio_data = np.array(audio.get_array_of_samples())
        sample_rate = audio.frame_rate
        return audio_data, sample_rate

    def passebas_filter_pydub(self, audio_data, sample_rate, cutoff_frequency:int = 2000):
        """
        Apply low pass filter to the audio data using pydub
        :param audio_data: Audio data
        :param sample_rate: Sample rate of the audio data
        :param cutoff_frequency: Cutoff frequency for the low pass filter
        :return: Filtered audio data
        """
        audio = AudioSegment(
            audio_data.tobytes(), 
            frame_rate=sample_rate,
            sample_width=audio_data.dtype.itemsize, 
            channels=1
        )
        filtered_audio = audio.low_pass_filter(cutoff_frequency)
        filtered_audio_data = np.array(filtered_audio.get_array_of_samples())
        return filtered_audio_data

    def passehaut_filter_pydub(self, audio_data, sample_rate, cutoff_frequency:int = 2000):
        """
        Apply high pass filter to the audio data using pydub
        :param audio_data: Audio data
        :param sample_rate: Sample rate of the audio data
        :param cutoff_frequency: Cutoff frequency for the high pass filter
        :return: Filtered audio data
        """
        audio = AudioSegment(
            audio_data.tobytes(), 
            frame_rate=sample_rate,
            sample_width=audio_data.dtype.itemsize, 
            channels=1
        )
        filtered_audio = audio.high_pass_filter(cutoff_frequency)
        filtered_audio_data = np.array(filtered_audio.get_array_of_samples())
        return filtered_audio_data
    

    def mel_spectrogram_pydub(self, audio_data, sample_rate, n_mels:int = 128):
        """
        Compute the mel spectrogram of the audio data using pydub
        :param audio_data: Audio data
        :param sample_rate: Sample rate of the audio data
        :param n_mels: Number of mel bands
        :return: Mel spectrogram
        """
        audio = AudioSegment(
            audio_data.tobytes(), 
            frame_rate=sample_rate,
            sample_width=audio_data.dtype.itemsize, 
            channels=1
        )
        samples = np.array(audio.get_array_of_samples())
        mel_spectrogram = librosa.feature.melspectrogram(y=samples, sr=sample_rate, n_mels=n_mels)
        return mel_spectrogram

    def mfcc_pydub(self, audio_data, sample_rate, n_mfcc:int = 13):
        """
        Compute the MFCC of the audio data using pydub
        :param audio_data: Audio data
        :param sample_rate: Sample rate of the audio data
        :param n_mfcc: Number of MFCC coefficients
        :return: MFCC
        """
        audio = AudioSegment(
            audio_data.tobytes(), 
            frame_rate=sample_rate,
            sample_width=audio_data.dtype.itemsize, 
            channels=1
        )
        samples = np.array(audio.get_array_of_samples())
        mfcc = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=n_mfcc)
        return mfcc
    
    def spectral_substraction_pydub(self, audio_data, sample_rate:int = 3200, n_fft:int = 2048, hop_length:int = 512, power:float = 1.0, margin:float = 1.0):
        """
        Apply spectral substraction to the audio data using pydub
        :param audio_data: Audio data
        :param sample_rate: Sample rate of the audio data
        :param n_fft: Number of FFT components
        :param hop_length: Hop length
        :param power: Power coefficient
        :param margin: Margin coefficient
        :return: Filtered audio data
        """
        audio = AudioSegment(
            audio_data.tobytes(), 
            frame_rate=sample_rate,
            sample_width=audio_data.dtype.itemsize, 
            channels=1
        )
        samples = np.array(audio.get_array_of_samples())
        stft = librosa.stft(samples, n_fft=n_fft, hop_length=hop_length)
        stft = np.abs(stft) ** power
        stft -= margin * np.mean(stft, axis=1, keepdims=True)
        stft[stft < 0] = 0
        filtered_audio_data = librosa.istft(stft, hop_length=hop_length)
        return filtered_audio_data
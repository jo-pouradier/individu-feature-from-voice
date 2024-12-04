from multiprocessing import process
import os
import librosa
import numpy as np
from pydub import AudioSegment

class Preprocessing:
    def __init__(self, samples_path:str, processed_path:str):
        self.samples_path = samples_path # Path to folder with audio files
        self.processed_path = processed_path # Path to folder to save processed audio files
        self.rate = 44100 


    def mp3_to_wav(self, clip_name) -> AudioSegment:
        clip_name = clip_name.split('.')[0]
        clip = AudioSegment.from_mp3(os.path.join(self.samples_path, f'{clip_name}.mp3'))
        return clip

    def normalize(self, clip:AudioSegment, base_clip:AudioSegment):
        clip = clip.set_frame_rate(self.rate)
        clip = clip.set_channels(1)
        clip = self.normalize_volume(clip, base_clip)
        return clip

    def normalize_volume(self, clip: AudioSegment | str, base_clip: AudioSegment | str) -> AudioSegment:
        diff = base_clip.dBFS - clip.dBFS
        clip = clip.apply_gain(diff)
        return clip

    def change_duration(self, clip: AudioSegment, duration: int = 10) -> AudioSegment:
        # set duration to 10 secondes
        # check audio duration
        duration_in_seconds = len(clip)/1000
        if len(clip)/1000 > duration:
            clip = clip[:duration*1000]
        else:
            silence_duration = (duration - duration_in_seconds) * 1000  # Convert to milliseconds
            silence = AudioSegment.silent(silence_duration)
            clip = clip + silence
        return clip

    def spectral_substraction_pydub(self, audio:AudioSegment, n_fft:int = 2048, hop_length:int = 512, power:float = 1.0, margin:float = 1.0):
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
        samples = np.array(audio.get_array_of_samples(), dtype=float)
        stft = librosa.stft(samples, n_fft=n_fft, hop_length=hop_length)
        stft = np.abs(stft) ** power
        stft -= margin * np.mean(stft, axis=1, keepdims=True)
        stft[stft < 0] = 0
        filtered_audio_data = librosa.istft(stft, hop_length=hop_length)
        return filtered_audio_data

    def mel_spectrogram_pydub(self, audio:AudioSegment, n_mels:int = 128):
        """
        Compute the mel spectrogram of the audio data using pydub
        :param audio_data: Audio data
        :param sample_rate: Sample rate of the audio data
        :param n_mels: Number of mel bands
        :return: Mel spectrogram
        """
        samples = np.array(audio.get_array_of_samples(), dtype=float)
        mel_spectrogram = librosa.feature.melspectrogram(y=samples, sr=audio.frame_rate, n_mels=n_mels)
        return mel_spectrogram

def preprocess_audio(path, base_path, samples_path, processed_path):
    new_audio_name = path.replace("mp3", "wav")
    new_path = os.path.join( processed_path, new_audio_name)
    process = Preprocessing(samples_path, processed_path)
    
    base_clip = process.mp3_to_wav(base_path)
    audio_processed = process.mp3_to_wav(path)
    audio_processed = process.normalize(audio_processed, base_clip=base_clip)
    audio_processed = process.change_duration(audio_processed, 10)
    # audio_processed.export(new_path, format='wav')
    spectro = process.mel_spectrogram_pydub(audio_processed)
    spectro = (spectro/np.max(np.abs(spectro)) * 127).astype(np.int8)
    np.save(new_path.replace("wav", "npy"), spectro)
    # audio_processed = process.spectral_substraction_pydub(audio_processed)
    # export np array to wav
    # np.save(new_path, audio_processed)
    # audio = AudioSegment(audio_processed, frame_rate=44100, sample_width=2, channels=1)
    # minify the audio file
    # audio.export(new_path, format='wav')

if __name__=="__main__":
    samples_path = os.path.join(os.getcwd(), "fr", "clips")
    processed_path = os.path.join(os.getcwd(), "fr", "processed")
    # path = list(os.listdir(samples_path))[0]
    base_path = list(os.listdir(samples_path))[0]
    for path in os.listdir(samples_path)[:10]:
        preprocess_audio(path, base_path, samples_path, processed_path)
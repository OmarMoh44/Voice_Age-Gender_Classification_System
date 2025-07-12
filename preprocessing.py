import os
import numpy as np
import soundfile as sf
import librosa
import warnings
import logging
from scipy import signal
from typing import List, Tuple, Optional
from joblib import Parallel, delayed
from tqdm import tqdm

class AudioPreprocessor:
    def __init__(self,
                 sample_rate: int = 16000,
                 vad_top_db: float = 30.0,
                 low_freq: float = 80.0,
                 high_freq: float = 4000.0,
                 pre_emphasis: float = 0.97):
        self.sample_rate = sample_rate
        self.vad_top_db = vad_top_db
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.pre_emphasis = pre_emphasis

        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=UserWarning)

    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        abs_path = os.path.abspath(file_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"File not found: {abs_path}")
        try:
            # Attempt with soundfile (better for WAV)
            audio, sr = sf.read(abs_path)
            # Convert stereo to mono (shape: [samples, channels] -> [samples])
            if audio.ndim > 1:
                audio = librosa.to_mono(audio.T)  # librosa expects [channels, samples]
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            return audio, self.sample_rate
        except Exception:
            # Fallback to librosa (handles MP3 and others)
            audio, sr = librosa.load(abs_path, sr=self.sample_rate, mono=True)
            return audio, self.sample_rate

    def bandpass_filter(self, audio: np.ndarray) -> np.ndarray:
        nyquist = self.sample_rate / 2.0
        low = self.low_freq / nyquist
        high = self.high_freq / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        return signal.filtfilt(b, a, audio)

    def remove_silence(self, audio: np.ndarray) -> np.ndarray:
        trimmed, _ = librosa.effects.trim(audio, top_db=self.vad_top_db)
        intervals = librosa.effects.split(trimmed, top_db=self.vad_top_db)
        return np.concatenate([trimmed[start:end] for start, end in intervals]) if intervals.any() else trimmed

    def normalize_amplitude(self, audio: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(audio)) + 1e-6
        return audio / peak

    def apply_preemphasis(self, audio: np.ndarray) -> np.ndarray:
        return np.append(audio[0], audio[1:] - self.pre_emphasis * audio[:-1])

    def process_audio(self, audio: np.ndarray) -> np.ndarray:
        audio = self.bandpass_filter(audio)
        audio = self.apply_preemphasis(audio)
        audio = self.remove_silence(audio)
        audio = self.normalize_amplitude(audio)
        return audio

    def process_file(self, file_path: str) -> Optional[np.ndarray]:
        try:
            audio, _ = self.load_audio(file_path)
            return self.process_audio(audio)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def process_and_save_file(self, source_folder: str, destination_folder: str, fname: str):
        src_path = os.path.join(source_folder, fname)
        processed_audio = self.process_file(src_path)
        if processed_audio is not None:
            dest_path = os.path.join(destination_folder, fname)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            sf.write(dest_path, processed_audio, self.sample_rate)

    def batch_process(self,
                      source_folder: str,
                      file_list: List[str],
                      destination_folder: str,
                      n_jobs: int = -1):
        Parallel(n_jobs=n_jobs)(
            delayed(self.process_and_save_file)(source_folder, destination_folder, fname)
            for fname in tqdm(file_list, desc="Parallel processing", total=len(file_list))
        )
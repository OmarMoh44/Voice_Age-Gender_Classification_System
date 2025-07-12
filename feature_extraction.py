import numpy as np
import librosa
import parselmouth
import os
from joblib import Parallel, delayed
from tqdm import tqdm

class AudioFeatureExtractor:
    def __init__(self, sr=16000, n_fft=1024, hop_length=256, lpc_order=12, n_mfcc=20):
        """
        Initialize the feature extractor with default parameters.
        
        Args:
            sr (int): Sample rate
            n_fft (int): FFT window size
            hop_length (int): Hop length for frame analysis
            lpc_order (int): Order for LPC analysis
            n_mfcc (int): Number of MFCC coefficients
        """
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.lpc_order = lpc_order
        self.n_mfcc = n_mfcc
        self.expected_feature_dim = 74  # Expected dimension of the final feature vector
        
        # Define feature groups for better analysis
        self.feature_groups = {
            'mfcc': (0, 40),  # 40 features (20 means + 20 stds)
            'vocal': (40, 46),  # 6 features
            'formants': (46, 58),  # 12 features
            'spectral': (58, 72),  # 14 features
            'cpp': (72, 74)  # 2 features
        }
        
        # Feature names for better interpretability
        self.feature_names = []
        # MFCC names
        for i in range(1, self.n_mfcc+1):
            self.feature_names.append(f'mfcc_{i}_mean')
        for i in range(1, self.n_mfcc+1):
            self.feature_names.append(f'mfcc_{i}_std')
        
        # Vocal feature names
        self.feature_names.extend(['pitch_mean', 'pitch_min', 'pitch_max', 'jitter', 'shimmer', 'hnr'])
        
        # Formant feature names
        self.feature_names.extend(['F1_mean', 'F2_mean', 'F3_mean', 
                                   'F1_std', 'F2_std', 'F3_std',
                                   'F2/F1_mean', 'F3/F1_mean', 'F3/F2_mean',
                                   'F2/F1_std', 'F3/F1_std', 'F3/F2_std'])
        
        # Spectral feature names
        self.feature_names.extend(['centroid_mean', 'centroid_std', 
                                   'bandwidth_mean', 'bandwidth_std',
                                   'rolloff_mean', 'rolloff_std',
                                   'tilt_mean', 'tilt_std',
                                   'flux_mean', 'flux_std',
                                   'alpha_ratio_mean', 'alpha_ratio_std',
                                   'hammarberg_mean', 'hammarberg_std'])
        
        # CPP feature names
        self.feature_names.extend(['cpp_mean', 'cpp_std'])

    def extract_mfccs(self, S_db):
        """Extract MFCC features from spectrogram."""
        mfccs = librosa.feature.mfcc(S=S_db, n_mfcc=self.n_mfcc)
        return np.concatenate([mfccs.mean(1), mfccs.std(1)])

    def extract_formants(self, audio):
        """Extract formant frequencies and ratios (mean and std only)."""
        try:
            audio = librosa.effects.preemphasis(audio)
            frames = librosa.util.frame(audio, frame_length=self.n_fft, hop_length=self.hop_length)
            formants, ratios = [], []
            
            for frame in frames.T:
                windowed = frame * np.hamming(self.n_fft)
                a = librosa.lpc(windowed, order=self.lpc_order)
                roots = np.roots(a)
                roots = roots[np.imag(roots) >= 0]
                
                # Filter valid roots
                mags = np.abs(roots)
                valid = (mags > 1e-10) & (mags < 1.0)
                roots = roots[valid]
                mags = mags[valid]
                
                # Convert to frequencies
                angles = np.arctan2(np.imag(roots), np.real(roots))
                freqs = angles * (self.sr / (2 * np.pi))
                bw = (-self.sr / np.pi) * np.log(mags)
                
                # Filter valid formants
                valid = (freqs > 50) & (freqs < 4000) & (bw > 0)
                freqs = freqs[valid]
                
                # Sort by frequency
                freqs = freqs[np.argsort(freqs)]
                
                # Pad to 3 formants if needed
                freqs_padded = np.pad(freqs[:3], (0, 3 - len(freqs[:3])), 'constant')
                formants.append(freqs_padded)
                
                # Calculate ratios
                ratio_frame = []
                if len(freqs) >= 2:
                    ratio_frame.append(freqs[1] / freqs[0])
                if len(freqs) >= 3:
                    ratio_frame.append(freqs[2] / freqs[0])
                    ratio_frame.append(freqs[2] / freqs[1])
                ratios.append(np.pad(ratio_frame, (0, 3 - len(ratio_frame)), 'constant'))
            
            formants = np.nan_to_num(np.array(formants), nan=0.0)
            ratios = np.nan_to_num(np.array(ratios), nan=0.0)

            return np.concatenate([
                formants.mean(0),  # Mean of F1, F2, F3
                formants.std(0),   # Std of F1, F2, F3
                ratios.mean(0),    # Mean of F2/F1, F3/F1, F3/F2
                ratios.std(0)      # Std of F2/F1, F3/F1, F3/F2
            ])
        except Exception as e:
            print(f"Error in extract_formants: {e}")
            return None
        
    def extract_spectral_features(self, S_mag):
        """Extract various spectral features."""
        try:
            freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)

            centroid = librosa.feature.spectral_centroid(S=S_mag)
            bandwidth = librosa.feature.spectral_bandwidth(S=S_mag)
            rolloff = librosa.feature.spectral_rolloff(S=S_mag)
            centroid_stats = np.array([centroid.mean(), centroid.std()])
            bandwidth_stats = np.array([bandwidth.mean(), bandwidth.std()])
            rolloff_stats = np.array([rolloff.mean(), rolloff.std()])

            tilts = []
            for frame in S_mag.T:
                coeffs = np.polyfit(np.arange(len(frame)), frame, 1)
                tilts.append(coeffs[0])
            tilt_feats = np.array([np.nanmean(tilts), np.nanstd(tilts)])

            flux = librosa.onset.onset_strength(S=S_mag)
            flux_stats = np.array([np.nanmean(flux), np.nanstd(flux)])

            mask_alpha_low = (freqs >= 50) & (freqs <= 1000)
            mask_alpha_high = (freqs > 1000) & (freqs <= 5000)
            alpha_low = np.sum(S_mag[mask_alpha_low, :], axis=0)
            alpha_high = np.sum(S_mag[mask_alpha_high, :], axis=0)
            alpha_ratio = alpha_low / (alpha_high + 1e-10)
            alpha_ratio_mean = np.nanmean(alpha_ratio)
            alpha_ratio_std = np.nanstd(alpha_ratio)

            mask_ham_low = (freqs >= 0) & (freqs <= 2000)
            mask_ham_high = (freqs > 2000) & (freqs <= 5000)
            max_ham_low = np.max(S_mag[mask_ham_low, :], axis=0)
            max_ham_high = np.max(S_mag[mask_ham_high, :], axis=0)
            hammarberg = max_ham_low / (max_ham_high + 1e-10)
            hammarberg_mean = np.nanmean(hammarberg)
            hammarberg_std = np.nanstd(hammarberg)

            return np.concatenate([
                centroid_stats,
                bandwidth_stats,
                rolloff_stats,
                tilt_feats,
                flux_stats,
                np.array([alpha_ratio_mean, alpha_ratio_std]),
                np.array([hammarberg_mean, hammarberg_std])
            ])
        except Exception as e:
            print(f"Error in extract_spectral_features: {e}")
            return None

    def extract_cpp(self, S_db):
        """Extract cepstral peak prominence features."""
        try:
            cpp = []
            for frame in S_db.T:
                cepstrum = np.fft.rfft(frame).real
                if len(cepstrum) < 50:
                    continue
                peak = np.max(cepstrum[10:50])
                trend = np.polyval(np.polyfit(np.arange(len(cepstrum)), cepstrum, 1), np.arange(len(cepstrum)))
                cpp.append(peak - np.mean(trend))
            return np.array([np.nanmean(cpp), np.nanstd(cpp)])
        except Exception:
            print("Error in extract_cpp")
            return None

    def extract_vocal_features(self, audio):
        """Extract vocal features using parselmouth (Praat)."""
        try:
            sound = parselmouth.Sound(audio, self.sr)
            time_step = self.hop_length / self.sr
            pitch = sound.to_pitch(time_step=time_step)
            f0 = pitch.selected_array['frequency']
            f0[f0 == 0] = np.nan

            pitch_mean = np.nanmean(f0)
            pitch_min = np.nanmin(f0)
            pitch_max = np.nanmax(f0)

            point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 500)
            jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            harmonicity = parselmouth.praat.call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = parselmouth.praat.call(harmonicity, "Get mean", 0, 0)

            return np.array([pitch_mean, pitch_min, pitch_max, jitter, shimmer, hnr])
        except Exception as e:
            print(f"Vocal Feature Error: {str(e)[:100]}")
            return None

    def extract_features_from_audio(self, audio_path=None, audio=None):
        """Extract all features from an audio file."""
        try:
            if audio is None:
                if audio_path is None:
                    raise ValueError("Either audio_path or audio must be provided.")
                audio, sr = librosa.load(audio_path, sr=self.sr)
            if len(audio) == 0:
                return None

            S = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
            S_mag, _ = librosa.magphase(S)
            S_db = librosa.amplitude_to_db(S_mag)

            features = [
                self.extract_mfccs(S_db), # 40 
                self.extract_vocal_features(audio), # 6  
                self.extract_formants(audio), #  12
                self.extract_spectral_features(S_mag), # 14 
                self.extract_cpp(S_db), # 2
            ]

            combined = np.concatenate([f for f in features if f is not None])
            if combined.shape != (self.expected_feature_dim,):
                print(f"Feature shape mismatch: {combined.shape} (expected {self.expected_feature_dim})")
                return None

            return combined
        except Exception as e:
            print(f"Error in extract_features_from_audio: {e}")
            return None

    def process_row(self, row, base_path):
        """Process a single row from a dataframe."""
        audio_path = os.path.join(base_path, row['path'])
        label = row["label"]

        try:
            feature_vec = self.extract_features_from_audio(audio_path)
            if feature_vec is None:
                return None, None, audio_path
            return feature_vec, label, None
        except Exception:
            return None, None, audio_path

    def process_dataset(self, df, base_path, x_out_path='X_all.npy', y_out_path='y_all.npy'):
        """Process an entire dataset in parallel."""
        rows = df.to_dict(orient='records')

        print("Extracting features...")

        results = Parallel(n_jobs=-1)(
            delayed(self.process_row)(row, base_path)
            for row in tqdm(rows, desc="Processing")
        )

        X, y, failed_paths = [], [], []

        for feature, label, failed in results:
            if failed:
                failed_paths.append(failed)
            elif feature is not None:
                X.append(feature)
                y.append(label)

        X = np.vstack(X)
        y = np.array(y)

        print(f"✅ Feature matrix shape: {X.shape}")
        print(f"✅ Labels shape: {y.shape}")
        print(f"❌ Failed paths: {len(failed_paths)}")

        np.save(x_out_path, X)
        np.save(y_out_path, y)

        return X, y, failed_paths
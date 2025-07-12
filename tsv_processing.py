import os
import pandas as pd
import warnings
import librosa
from joblib import Parallel, delayed
from tqdm import tqdm


class AudioTSVProcessor:
    def __init__(self, tsv_file: str, base_path: str, filtered_path: str, sr: int = 16000):
        self.tsv_file = tsv_file
        self.base_path = base_path
        self.filtered_path = filtered_path
        self.sample_rate = sr
        self.valid_rows = []
        self.corrupted_paths = []

    def load_dataframe(self):
        self.df = pd.read_csv(self.tsv_file, sep="\t", dtype=str)
        self.rows = self.df.to_dict(orient="records")

    def check_audio_row(self, row_dict: dict):
        audio_path = os.path.join(self.base_path, row_dict["path"])
        
        if not os.path.isfile(audio_path):
            return None, audio_path

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                librosa.load(audio_path, sr=self.sample_rate)
            return row_dict, None
        except Exception:
            return None, audio_path

    def validate_audio_files(self):
        print("Checking audio files...")
        results = Parallel(n_jobs=-1)(
            delayed(self.check_audio_row)(row)
            for row in tqdm(self.rows, desc="Validating")
        )

        self.valid_rows = [row for row, err in results if row is not None]
        self.corrupted_paths = [err for row, err in results if err is not None]

    def save_filtered_dataframe(self):
        new_df = pd.DataFrame(self.valid_rows)
        new_df.to_csv(self.filtered_path, sep="\t", index=False)
        return new_df

    def process(self):
        self.load_dataframe()
        self.validate_audio_files()
        print(f"\n✅ {len(self.valid_rows)} valid audio files")
        print(f"❌ {len(self.corrupted_paths)} corrupted or unreadable files")
        return self.save_filtered_dataframe()

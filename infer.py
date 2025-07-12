import os
import argparse
import joblib
import numpy as np
import time
from natsort import natsorted
from preprocessing import AudioPreprocessor
from feature_extraction import AudioFeatureExtractor

def load_artifacts(model_path='models/model.joblib'):
    """Load trained model and feature indices."""
    model = joblib.load(model_path)
    selected_indices = np.load('models/selected_indices.npy')
    return model, selected_indices

def process_files(data_dir, model, selected_indices, scaler):
    """Process files in natural sorted order."""
    num = 0
    files = natsorted(os.listdir(data_dir))
    preprocessor = AudioPreprocessor(sample_rate=16000)
    extractor = AudioFeatureExtractor(sr=16000)
    predictions = []

    for filename in files:
        file_path = os.path.join(data_dir, filename)
        try:
            processed_audio = preprocessor.process_file(file_path)
            features = extractor.extract_features_from_audio(audio=processed_audio)
            features = np.nan_to_num(features, nan=0.0)
            features_norm = scaler.transform(features.reshape(1, -1))

            if features_norm is None or features_norm.shape != (1, 74):
                raise ValueError("Invalid features")

            selected_features_norm = features_norm[0, selected_indices]
            pred = model.predict([selected_features_norm])[0]
            predictions.append(str(pred))
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            predictions.append("error")
        print(num)
        num += 1

    return predictions

def save_results(predictions, path='results.txt'):
    """Save predictions in required format."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(predictions))
    print(f"✅ Results saved to {path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to directory with audio files')
    parser.add_argument('--model_path', type=str, default='models/model.joblib',
                        help='Path to the model file')
    parser.add_argument('--save_path', type=str, default='result/results.txt',
                        help='Path to save results')
    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        raise ValueError(f"Directory {args.data_dir} does not exist")

    model, selected_indices = load_artifacts(model_path=args.model_path)
    scaler = joblib.load('models/scaler.joblib')

    print("Starting inference...")
    start_time = time.time()
    predictions = process_files(args.data_dir, model, selected_indices, scaler)
    end_time = time.time()
    print(f"✅ Inference completed in {end_time - start_time:.3f} seconds")

    save_results(predictions, args.save_path)

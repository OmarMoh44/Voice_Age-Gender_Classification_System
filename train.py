from tsv_processing import AudioTSVProcessor
from preprocessing import AudioPreprocessor
from feature_selection import FeatureAnalyzer
from feature_extraction import AudioFeatureExtractor
from model_selection import ModelTrainer

import numpy as np
import pandas as pd
import joblib


if __name__ == "__main__":
    audios_path = "audios"
    preprocessed_path = "preprocessed"
    old_tsv_file_path = "filtered_data_labeled.tsv"
    new_tsv_file_path = "new_filtered_data_labeled.tsv"
    
    # Step 1: Process and filter the TSV file
    processor = AudioTSVProcessor(
        tsv_file=old_tsv_file_path,
        base_path=audios_path,
        filtered_path=new_tsv_file_path,
    )
    filtered_df = processor.process()
    
    # Step 2: Preprocess audio files
    paths = filtered_df['path'].tolist()
    preprocessor = AudioPreprocessor(sample_rate=16000)
    preprocessor.batch_process(
        source_folder=audios_path,
        file_list=paths,
        destination_folder=preprocessed_path,
        n_jobs=-1 
    )

    # Step 3: Extract audio features
    extractor = AudioFeatureExtractor(sr=16000)
    X, y, failed = extractor.process_dataset(filtered_df, preprocessed_path,'X.npy', 'y.npy')
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]
    
    # Step 4: Analyze features and standardize them
    analyzer = FeatureAnalyzer(extractor)
    X_selected, y, selected_indices = analyzer.run_full_analysis(X, y)

    # Step 5: Train on different models and select the best one
    trainer = ModelTrainer(X_selected, y, test_size=0.2, random_state=42)
    trainer.enable_smote(sampling_strategy='auto', k_neighbors=5)
    trainer.enable_class_weights(weight_strategy='balanced')
    trainer.train_models(cv=3)  # 3-fold cross-validation, disable random search 
    trainer.print_results()
    best_name, best_model, best_score = trainer.get_best_model(metrix='test_accuracy')
    print(f"\nBest model: {best_name} with accuracy: {best_score:.4f}")
    
    # Step 6: Save the best model to disk
    joblib.dump(best_model, 'models/model.joblib')
    print(f"Saved best model ({best_name}) to models/model.joblib")
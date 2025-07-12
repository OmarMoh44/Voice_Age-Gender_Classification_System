# Voice Age/Gender Classification System  

## Project Overview  
This system classifies voice recordings into 4 categories:  
- **0**: Male (20-35 years)  
- **1**: Female (20-35 years)  
- **2**: Male (50-65 years)  
- **3**: Female (50-65 years)  

The pipeline includes:  
1. **TSV Processing**: Filter and validate audio files from TSV dataset
2. **Preprocessing**: Noise removal, silence trimming, bandpass filtering, normalization  
3. **Feature Extraction**: 74 acoustic features (MFCCs, formants, pitch, spectral features, CPP)  
4. **Feature Selection**: Reduce feature vector to 55 optimal features using Random Forest importance
5. **Model Training**: Evaluate multiple classifiers with SMOTE and class balancing, select best performer  

---

## Requirements and Installation   
- Python 3.8+  
- System dependencies: `libsndfile1 ffmpeg`  
- Python packages (install via):  
  ```bash
  pip install -r requirements.txt
  ```

---

## Project Structure
```
Team11/
├── models/                          # Trained model artifacts
│   ├── model.joblib               # Best trained classifier
│   ├── scaler.joblib              # Feature standardization scaler
│   └── selected_indices.npy       # Selected feature indices
├── feature anaysis/                # Feature analysis visualizations
│   ├── Feature Importance.jpg     # Feature importance plot
│   ├── Feature Correlation.jpg    # Feature correlation matrix
│   ├── Group Importance.jpg       # Feature group performance
│   └── Confusion Matrix.jpg       # Model confusion matrix
├── tsv_processing.py              # TSV dataset processing
├── preprocessing.py                # Audio preprocessing pipeline
├── feature_extraction.py          # 74 acoustic feature extraction
├── feature_selection.py           # Feature analysis and selection
├── model_selection.py             # Multi-model training and evaluation
├── train.py                       # Complete training pipeline
├── infer.py                       # Inference script
├── requirements.txt               # Python dependencies
├── dockerfile                     # Docker configuration
└── README.md                     # This file
```

---

## Feature Engineering

### Audio Preprocessing
- **Sample Rate**: 16kHz standardization
- **Bandpass Filter**: 80Hz - 4kHz frequency range
- **Silence Removal**: Voice Activity Detection (VAD) with 30dB threshold
- **Amplitude Normalization**: Peak normalization
- **Pre-emphasis**: 0.97 coefficient for high-frequency enhancement

### Feature Extraction (74 Features)
1. **MFCC Features (40)**: 20 coefficients × 2 (mean, std)
2. **Vocal Features (6)**: Pitch (mean, min, max), Jitter, Shimmer, HNR
3. **Formant Features (12)**: F1, F2, F3 (mean, std) + ratios (F2/F1, F3/F1, F3/F2)
4. **Spectral Features (14)**: Centroid, Bandwidth, Rolloff, Tilt, Flux, Alpha Ratio, Hammarberg Index
5. **CPP Features (2)**: Cepstral Peak Prominence (mean, std)

### Feature Selection
- **Method**: Random Forest importance-based selection
- **Target**: 55 optimal features from 74 total features
- **Analysis**: Feature correlation, group importance, and individual feature importance

---

## Model Training

### Supported Classifiers
- **Logistic Regression**: Linear classification with regularization
- **Random Forest**: Ensemble of 200 decision trees
- **SVM**: Support Vector Machine with multiple kernels
- **K-Nearest Neighbors**: Distance-based classification
- **XGBoost**: Gradient boosting with optimized parameters

### Training Features
- **Cross-Validation**: 3-fold cross-validation for model selection
- **SMOTE**: Synthetic Minority Over-sampling for class balance
- **Class Weights**: Balanced class weights for imbalanced datasets
- **Hyperparameter Tuning**: RandomizedSearchCV for optimal parameters

---

## Usage

### Training the Model
```bash
python train.py
```

**Note**: Update the following paths in `train.py`:
- `audios_path`: Directory containing audio files
- `old_tsv_file_path`: Path to original TSV dataset file
- `new_tsv_file_path`: Path for filtered TSV file

### Running Inference
```bash
python infer.py --data_dir <directory_of_test_files> --model_path <path_of_model> --save_path <output_path>
```

**Parameters**:
- `--data_dir`: Directory containing audio files to classify (required)
- `--model_path`: Path to trained model (default: `models/model.joblib`)
- `--save_path`: Path to save results (default: `result/results.txt`)

**Example**:
```bash
python infer.py --data_dir test_audio/ --save_path predictions.txt
```

---

## Docker Usage

### Pull the Docker Image
```bash
docker pull omar1101/team11
```

### Run the Docker Container
```bash
docker run \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  omar1101/team11 \
  --data_dir /app/data \
  --save_path /app/output/results.txt
```

**Volume Mounts**:
- `-v $(pwd)/data:/app/data`: Mounts your local data directory to `/app/data` in the container
- `-v $(pwd)/output:/app/output`: Mounts your local output directory to `/app/output` in the container

**Parameters**:
- `--data_dir /app/data`: Specifies the input directory containing audio files
- `--save_path /app/output/results.txt`: Specifies the path to save the prediction results

---

## Output Format

The inference script generates a text file with one prediction per line, corresponding to the input audio files in natural sorted order:

```
0
1
2
3
...
```

Where each number represents:
- **0**: Male (20-35 years)
- **1**: Female (20-35 years)  
- **2**: Male (50-65 years)
- **3**: Female (50-65 years)

---

## Performance Analysis

The system includes comprehensive feature and model analysis:

### Feature Analysis
- **Feature Importance**: Random Forest-based feature ranking
- **Feature Correlation**: Correlation matrix visualization
- **Group Importance**: Performance analysis by feature groups
- **Feature Selection**: Optimal feature subset identification

### Model Evaluation
- **Cross-Validation**: Robust performance estimation
- **Confusion Matrix**: Detailed classification performance
- **Class Distribution**: Dataset balance analysis
- **Multiple Metrics**: Accuracy, precision, recall, F1-score

---

## Dependencies

### Core Libraries
- `numpy`: Numerical computing
- `librosa`: Audio processing and feature extraction
- `praat-parselmouth`: Voice analysis (pitch, formants)
- `scikit-learn`: Machine learning algorithms
- `xgboost`: Gradient boosting classifier
- `imblearn`: SMOTE for class balancing

### Visualization
- `matplotlib`: Plotting and visualization
- `seaborn`: Statistical data visualization

### Utilities
- `joblib`: Model persistence and parallel processing
- `tqdm`: Progress bars
- `pandas`: Data manipulation
- `soundfile`: Audio file I/O
- `natsort`: Natural sorting for file processing

    
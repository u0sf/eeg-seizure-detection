# Implementation Plan: Bonn EEG Seizure Detection

## Goal
Implement a clean, educational, and professional Python project for epileptic seizure detection using the University of Bonn EEG dataset. The project will compare Classical ML baselines with Deep Learning approaches (1D CNN and 2D CNN).

## User Review Required
> [!NOTE]
> I will use **PyTorch** as the deep learning framework.
> I will assume the data is located in `data/bonn/` with subfolders `Z`, `O`, `N`, `F`, `S` (standard naming) or `A`, `B`, `C`, `D`, `E`. I will handle both naming conventions.
> I will include a **synthetic data generator** so the code can be run immediately for demonstration without needing to manually download the dataset first.

## Proposed Structure

### Directory Layout
```
eeg_seizure_detection/
├── data/                  # Data storage
│   └── bonn/              # Dataset files (A/Z, B/O, etc.)
├── src/                   # Source code
│   ├── __init__.py
│   ├── config.py          # Configuration (paths, params)
│   ├── data_loader.py     # Data reading & Dataset classes
│   ├── preprocessing.py   # Filtering, Normalization
│   ├── features.py        # Feature extraction (Handcrafted, Spectrograms)
│   ├── models.py          # PyTorch Models (1D CNN, 2D CNN)
│   ├── train.py           # Training loops
│   └── utils.py           # Metrics, Plotting, Seeding
├── notebooks/
│   └── analysis.ipynb     # Main entry point: EDA, Training, Evaluation
├── requirements.txt
└── README.md              # Project documentation
```

### Components

#### [NEW] src/config.py
- Define constants: Sampling rate (173.61 Hz), Class mappings, Paths.

#### [NEW] src/data_loader.py
- `load_bonn_data(path)`: Reads text files, handles the 5 subsets.
- `BonnDataset(Dataset)`: PyTorch Dataset wrapper.
- `generate_synthetic_data()`: For testing without the real dataset.

#### [NEW] src/preprocessing.py
- `bandpass_filter(signal, low, high, fs)`: Butterworth filter.
- `normalize_signal(signal)`: Z-score normalization per segment.

#### [NEW] src/features.py
- `extract_handcrafted_features(signal)`: Mean, std, skew, kurtosis, PSD band power.
- `compute_spectrogram(signal)`: STFT -> Log-Spectrogram -> Resize.

#### [NEW] src/models.py
- `Simple1DCNN`: 3 Conv blocks + FC.
- `Simple2DCNN`: For spectrograms.
- `ClassicalClassifier`: Wrapper for Scikit-Learn (RF/SVM).

#### [NEW] src/train.py
- `train_model()`: Training loop with Early Stopping.
- `evaluate_model()`: Test set evaluation.

#### [NEW] src/utils.py
- `set_seed()`: Reproducibility.
- `plot_confusion_matrix()`
- `plot_roc_curve()`
- `plot_signals()`

#### [NEW] notebooks/analysis.ipynb
- Step-by-step execution:
    1. Setup & Data Loading (Real or Synthetic).
    2. EDA (Plots).
    3. Preprocessing Demo.
    4. Classical ML Baseline (RF/SVM).
    5. Deep Learning (1D CNN).
    6. Deep Learning (2D CNN).
    7. Comparison & Conclusion.

## Verification Plan
1. **Synthetic Run**: Run the full pipeline with synthetic data to ensure no crashes.
2. **Shape Checks**: Verify tensor shapes at each stage (Preprocessing -> Model -> Output).
3. **Metric Sanity**: Check if metrics are calculated correctly (e.g., Accuracy between 0 and 1).

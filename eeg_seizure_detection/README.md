# Bonn EEG Seizure Detection

A professional, educational Python project for epileptic seizure detection using the University of Bonn EEG dataset.

## Project Structure

```
eeg_seizure_detection/
├── data/                  # Data storage (Bonn dataset)
├── src/                   # Source code
│   ├── config.py          # Configuration
│   ├── data_loader.py     # Data loading & Synthetic generation
│   ├── preprocessing.py   # Filtering & Normalization
│   ├── features.py        # Feature extraction
│   ├── models.py          # PyTorch Models & Classical ML
│   ├── train.py           # Training loops
│   └── utils.py           # Helper functions
├── artifacts/             # Saved models and plots
├── main.py                # Main execution script
└── requirements.txt       # Dependencies
```

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install numpy pandas matplotlib seaborn scipy scikit-learn torch torchvision
    ```

2.  **Data:**
    *   The project is configured to look for the Bonn dataset in `data/bonn/`.
    *   If the data is not found, it will automatically generate **synthetic data** for demonstration purposes.
    *   To use real data, place the folders (Z, O, N, F, S) inside `data/bonn/`.

## Usage

Run the main script to execute the full pipeline:

```bash
python eeg_seizure_detection/main.py
```

This will:
1.  Load data (or generate synthetic).
2.  Preprocess signals (Bandpass filter, Z-score normalization).
3.  Extract features (Handcrafted & Spectrograms).
4.  Train and evaluate a **Classical ML Baseline** (Random Forest).
5.  Train and evaluate a **1D CNN** (Deep Learning).
6.  Train and evaluate a **2D CNN** (Deep Learning).
7.  Save plots and models to `artifacts/`.

## Models

*   **Classical ML:** Random Forest using statistical features (Mean, Std, Skew, Kurtosis, PSD Band Power).
*   **1D CNN:** 3-layer Convolutional Neural Network operating on raw time-series data.
*   **2D CNN:** 3-layer Convolutional Neural Network operating on Spectrogram images.

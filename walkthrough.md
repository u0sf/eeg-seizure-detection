# Walkthrough: Bonn EEG Seizure Detection

I have implemented a complete, modular Python project for epileptic seizure detection.

## 1. Project Overview
The project is organized into a clean structure in `eeg_seizure_detection/`:
*   **`src/`**: Contains all the logic (data loading, preprocessing, models, training).
*   **`main.py`**: The entry point to run the entire pipeline.
*   **`README.md`**: Instructions on how to run the code.
*   **`REPORT.md`**: A university-style report describing the methods and results.

## 2. Key Features
*   **Synthetic Data Generator**: If you don't have the Bonn dataset downloaded yet, the code automatically generates synthetic EEG-like data so you can run and test the pipeline immediately.
*   **Multiple Models**:
    *   **Classical ML**: Random Forest with handcrafted features.
    *   **1D CNN**: Deep learning on raw signals.
    *   **2D CNN**: Deep learning on spectrograms.
*   **Visualization**: The pipeline automatically saves plots (Confusion Matrices, ROC Curves, Training History, Raw Signals) to the `artifacts/` folder.

## 3. How to Run
Open a terminal and run:
```bash
python eeg_seizure_detection/main.py
```

## 4. Next Steps
1.  Download the real Bonn dataset (Sets A-E).
2.  Place the folders inside `eeg_seizure_detection/data/bonn/`.
3.  Re-run `main.py` to see real results.

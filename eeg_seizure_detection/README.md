# EEG Seizure Detection

This project uses Deep Learning (1D CNN and 2D CNN) to detect epileptic seizures from EEG signals using the Bonn University Dataset.

## 📂 Project Structure

- `src/`: Source code for data loading, preprocessing, feature extraction, and models.
- `main.py`: Script for training and evaluating the models.
- `gui_app.py`: Streamlit application for real-time inference.
- `data/`: Directory for the dataset (not included in repo).
- `artifacts/`: Directory for saved models and plots.

## 🚀 How to Run Locally

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Train the Model:**
    ```bash
    python main.py
    ```

3.  **Run the GUI App:**
    ```bash
    streamlit run gui_app.py
    ```

## 🌐 Deployment on Streamlit Cloud

You can easily deploy this app on Streamlit Cloud to share it with others:

1.  **Fork/Push this repository** to your GitHub account.
2.  Go to [Streamlit Cloud](https://streamlit.io/cloud) and sign in.
3.  Click **"New app"**.
4.  Select your repository (`eeg-seizure-detection`), branch (`main`), and main file path (`gui_app.py`).
5.  Click **"Deploy"**.

Streamlit Cloud will automatically install the dependencies from `requirements.txt` and launch the app.

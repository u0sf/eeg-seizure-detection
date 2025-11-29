import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add src to path so we can import modules
# This assumes gui_app.py is in the eeg_seizure_detection folder
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Try to import from src, handle potential path issues
try:
    from src.models import Simple1DCNN, Simple2DCNN
    from src.preprocessing import preprocess_pipeline
    from src.features import compute_spectrogram
    from src.config import N_SAMPLES, SAMPLING_RATE
except ImportError:
    # Fallback if run from root directory
    sys.path.append(os.path.join(os.path.dirname(__file__), 'eeg_seizure_detection', 'src'))
    from src.models import Simple1DCNN, Simple2DCNN
    from src.preprocessing import preprocess_pipeline
    from src.features import compute_spectrogram
    from src.config import N_SAMPLES, SAMPLING_RATE

# Page Config
st.set_page_config(
    page_title="EEG Seizure Detection",
    page_icon="🧠",
    layout="centered"
)

# Title and Description
st.title("🧠 EEG Seizure Detection (Bonn Dataset)")
st.markdown("""
This application uses a Deep Learning model (**1D CNN**) to analyze EEG signals and detect epileptic seizures.
Upload a single-channel EEG file (`.txt`) to get a prediction.
""")

# Sidebar
st.sidebar.header("About")
st.sidebar.info("This model was trained on the University of Bonn EEG dataset.")
st.sidebar.markdown("---")
st.sidebar.write("Developed with Streamlit & PyTorch")

# Model Loading
@st.cache_resource
def load_model():
    """Loads the trained model (1D or 2D)."""
    device = torch.device("cpu") # Inference on CPU
    
    # Path to the saved model weights
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # Check common locations
    possible_paths = [
        os.path.join(base_dir, 'artifacts', 'best_model_1d.pth'),
        os.path.join(os.path.dirname(base_dir), 'artifacts', 'best_model_1d.pth'),
        os.path.join(base_dir, 'artifacts', 'best_model.pth'),
        os.path.join(os.path.dirname(base_dir), 'artifacts', 'best_model.pth')
    ]
    
    model_path = None
    for p in possible_paths:
        if os.path.exists(p):
            model_path = p
            break
            
    if model_path is None:
        st.error("Model file not found in `artifacts/` directory.")
        st.warning("Please train the model first using `main.py`.")
        st.stop()
        
    try:
        # Try 1D first
        model = Simple1DCNN(num_classes=2)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, '1d'
    except RuntimeError:
        # Try 2D
        try:
            model = Simple2DCNN(num_classes=2)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            return model, '2d'
        except Exception as e:
             st.error(f"Error loading model as 2D: {e}")
             st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model, model_type = load_model()

# File Uploader
uploaded_file = st.file_uploader("Upload EEG Signal (.txt)", type=["txt"])

def load_signal(uploaded_file):
    """Reads the uploaded text file."""
    try:
        # Read file as string, then convert to numpy array
        content = uploaded_file.read().decode("utf-8")
        # Handle different delimiters just in case (Bonn is usually newlines)
        signal = np.fromstring(content, sep='\n')
        if len(signal) == 0:
             # Try space delimiter
             signal = np.fromstring(content, sep=' ')
        return signal
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

if uploaded_file is not None:
    # Load Signal
    signal = load_signal(uploaded_file)
    
    if signal is not None and len(signal) > 0:
        # Check length
        if len(signal) != N_SAMPLES:
            st.warning(f"Warning: Signal length is {len(signal)}, expected {N_SAMPLES}. It will be padded or trimmed.")
            # Resize
            if len(signal) > N_SAMPLES:
                signal = signal[:N_SAMPLES]
            else:
                signal = np.pad(signal, (0, N_SAMPLES - len(signal)), 'constant')
        
        # Plot Raw Signal
        st.subheader("Raw EEG Signal")
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(signal, color='blue', linewidth=0.5)
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Amplitude")
        ax.set_title("Input Signal")
        st.pyplot(fig)
        
        # Analyze Button
        if st.button("Analyze Signal", type="primary"):
            with st.spinner("Analyzing..."):
                # Preprocess
                if model_type == '1d':
                    # Preprocess for 1D: Filter + Normalize
                    processed_signal = preprocess_pipeline(signal, apply_filter=True, apply_norm=True)
                    # Input shape: (1, 1, 4097)
                    input_tensor = torch.FloatTensor(processed_signal).unsqueeze(0).unsqueeze(0)
                else:
                    # Preprocess for 2D: Filter (no norm) -> Spectrogram
                    sig_filt = preprocess_pipeline(signal, apply_filter=True, apply_norm=False)
                    spec = compute_spectrogram(sig_filt)
                    # Input shape: (1, 1, H, W)
                    input_tensor = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0)
                
                # Prediction
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)
                    confidence, predicted_class = torch.max(probs, 1)
                    
                # Results
                # TODO: Verify class mapping matches your training (0=Non-Seizure, 1=Seizure)
                label_map = {0: "Non-Seizure", 1: "Seizure"}
                prediction_label = label_map[predicted_class.item()]
                confidence_score = confidence.item() * 100
                
                # Display Results
                st.divider()
                st.subheader("Prediction Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if predicted_class.item() == 1:
                        st.error(f"**Prediction:** {prediction_label}")
                    else:
                        st.success(f"**Prediction:** {prediction_label}")
                        
                with col2:
                    st.metric("Confidence", f"{confidence_score:.2f}%")
                
                # Detailed Probabilities
                st.write("### Class Probabilities")
                prob_dict = {
                    "Non-Seizure": probs[0][0].item(),
                    "Seizure": probs[0][1].item()
                }
                st.bar_chart(prob_dict)
    else:
        st.error("Could not read valid signal data from file.")

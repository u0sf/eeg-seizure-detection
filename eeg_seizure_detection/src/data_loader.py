import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from .config import DATA_DIR, N_SAMPLES, CLASS_MAPPING_5, SAMPLING_RATE

def load_bonn_data(data_dir=DATA_DIR, allow_synthetic=False):
    """
    Loads the Bonn EEG dataset from text files.
    Expected structure: data_dir/{Z,O,N,F,S}/*.txt
    
    Returns:
        X (np.array): Signals (n_samples, 4097)
        y (np.array): Labels (n_samples,)
    """
    X = []
    y = []
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        if allow_synthetic:
            print(f"Data directory {data_dir} not found. Using synthetic data.")
            return generate_synthetic_data()
        else:
            raise FileNotFoundError(f"Data directory {data_dir} not found. Please download the Bonn dataset or use --use-synthetic.")

    # Map folder names to classes
    # We look for folders starting with the key letters
    found_data = False
    
    # Standard folder names in Bonn dataset are often just the letters or 'Set A', etc.
    # We will search for subdirectories
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if not subdirs:
         if allow_synthetic:
             print(f"No subdirectories found in {data_dir}. Using synthetic data.")
             return generate_synthetic_data()
         else:
             raise FileNotFoundError(f"No subdirectories found in {data_dir}. Please check your data structure.")

    print(f"Found subdirectories: {subdirs}")
    
    for subdir in subdirs:
        # Determine class based on folder name
        # Heuristic: check if folder name contains the class letter
        label = None
        for key, val in CLASS_MAPPING_5.items():
            # Check if the key (e.g., 'Z') is in the subdir name (e.g., 'Z' or 'Set Z')
            # We use a simple check. 
            # Note: 'A' and 'Z' are the same class, so we prioritize the Mapping values.
            # Actually, let's just check the first letter or if it matches known sets.
            if subdir.upper().startswith(key) or subdir.upper() == key:
                label = val
                break
        
        if label is None:
            continue
            
        folder_path = os.path.join(data_dir, subdir)
        files = glob.glob(os.path.join(folder_path, '*.txt'))
        
        for fpath in files:
            try:
                # Read text file (single column of numbers)
                signal = np.loadtxt(fpath)
                if len(signal) == N_SAMPLES:
                    X.append(signal)
                    y.append(label)
                else:
                    # Handle cases where length might be slightly different (though Bonn is usually consistent)
                    # For simplicity, we skip or trim/pad. Let's skip for now to be safe.
                    pass
            except Exception as e:
                print(f"Error reading {fpath}: {e}")
                
    if len(X) == 0:
        print("No valid data found. Using synthetic data.")
        return generate_synthetic_data()
        
    X = np.array(X)
    y = np.array(y)
    print(f"Loaded {len(X)} segments from {data_dir}")
    return X, y

def generate_synthetic_data(n_samples_per_class=100):
    """
    Generates synthetic EEG-like data for demonstration.
    5 Classes:
    0, 1: Healthy (Low freq, low amp)
    2, 3: Interictal (Medium freq, spikes)
    4: Ictal (High freq, high amp, chaotic)
    """
    print("Generating synthetic data...")
    X = []
    y = []
    
    t = np.linspace(0, N_SAMPLES/SAMPLING_RATE, N_SAMPLES)
    
    for label in range(5):
        for _ in range(n_samples_per_class):
            noise = np.random.normal(0, 0.5, N_SAMPLES)
            
            if label in [0, 1]: # Healthy
                # Alpha/Beta waves (8-30Hz)
                freq = np.random.uniform(8, 15)
                amp = np.random.uniform(10, 30)
                signal = amp * np.sin(2 * np.pi * freq * t) + noise
                
            elif label in [2, 3]: # Interictal
                # Slower waves + occasional spikes
                freq = np.random.uniform(4, 8) # Theta
                amp = np.random.uniform(30, 80)
                signal = amp * np.sin(2 * np.pi * freq * t)
                # Add spikes
                n_spikes = np.random.randint(1, 5)
                for _ in range(n_spikes):
                    spike_idx = np.random.randint(0, N_SAMPLES)
                    signal[spike_idx:min(spike_idx+10, N_SAMPLES)] += 200
                signal += noise
                
            elif label == 4: # Ictal (Seizure)
                # High frequency, high amplitude
                freq = np.random.uniform(15, 30) # Beta/Gamma
                amp = np.random.uniform(100, 300)
                # Frequency modulation to simulate seizure evolution
                signal = amp * np.sin(2 * np.pi * freq * t + np.sin(2*np.pi*0.5*t)) + noise * 5
                
            X.append(signal)
            y.append(label)
            
    return np.array(X), np.array(y)

class BonnDataset(Dataset):
    """
    PyTorch Dataset for Bonn EEG.
    """
    def __init__(self, X, y, transform=None, target_transform=None):
        """
        Args:
            X (np.array): Signals
            y (np.array): Labels
            transform (callable): Transform to apply to signal (e.g. spectrogram)
            target_transform (callable): Transform to apply to label (e.g. binary mapping)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.transform:
            x = self.transform(x)
            
        if self.target_transform:
            y = self.target_transform(y)
            
        return x, y

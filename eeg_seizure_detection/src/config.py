import os

# Project Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'bonn')
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts') # For saving models/plots

# Create artifacts dir if it doesn't exist
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# EEG Parameters
SAMPLING_RATE = 173.61  # Hz
SEGMENT_DURATION = 23.6 # seconds
N_SAMPLES = 4097        # per segment

# Dataset Mappings
# The Bonn dataset typically has folders Z, O, N, F, S (or A, B, C, D, E)
# A/Z: Healthy, Eyes Open
# B/O: Healthy, Eyes Closed
# C/N: Epileptic, Interictal (Hippocampal formation)
# D/F: Epileptic, Interictal (Epileptogenic zone)
# E/S: Epileptic, Ictal (Seizure)

CLASS_MAPPING_5 = {
    'A': 0, 'Z': 0,
    'B': 1, 'O': 1,
    'C': 2, 'N': 2,
    'D': 3, 'F': 3,
    'E': 4, 'S': 4
}

# Binary Mapping: Non-Seizure (0-3) vs Seizure (4)
# We will handle this mapping logic in the data loader
BINARY_CLASS_NAMES = ['Non-Seizure', 'Seizure']

# Preprocessing
FILTER_LOW_CUT = 0.5
FILTER_HIGH_CUT = 60.0 # Slightly below Nyquist (approx 86Hz), 60Hz is safe and removes high freq noise

# Training
SEED = 42
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
TEST_SIZE = 0.2
VAL_SIZE = 0.2 # of the remaining 80% (so 0.8 * 0.2 = 16% of total)

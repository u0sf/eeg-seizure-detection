import numpy as np
from scipy.signal import butter, filtfilt
from .config import SAMPLING_RATE, FILTER_LOW_CUT, FILTER_HIGH_CUT

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Creates Butterworth bandpass filter coefficients."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut=FILTER_LOW_CUT, highcut=FILTER_HIGH_CUT, fs=SAMPLING_RATE, order=5):
    """
    Applies a Butterworth bandpass filter to the data.
    
    Args:
        data (np.array): 1D EEG signal.
        lowcut (float): Lower frequency bound.
        highcut (float): Upper frequency bound.
        fs (float): Sampling rate.
        order (int): Filter order.
        
    Returns:
        np.array: Filtered signal.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def normalize_signal(data):
    """
    Applies Z-score normalization to the signal (per segment).
    x_norm = (x - mean) / std
    
    Args:
        data (np.array): 1D EEG signal.
        
    Returns:
        np.array: Normalized signal.
    """
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return data - mean # Avoid division by zero
    return (data - mean) / std

def preprocess_pipeline(signal, apply_filter=True, apply_norm=True):
    """
    Full preprocessing pipeline.
    """
    processed = signal
    if apply_filter:
        processed = apply_bandpass_filter(processed)
    if apply_norm:
        processed = normalize_signal(processed)
    return processed

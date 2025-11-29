import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch, spectrogram
from scipy.ndimage import zoom
from .config import SAMPLING_RATE

def extract_handcrafted_features(signal, fs=SAMPLING_RATE):
    """
    Extracts statistical and frequency domain features.
    
    Features:
    - Mean, Std, Skewness, Kurtosis
    - Band power (Delta, Theta, Alpha, Beta, Gamma)
    """
    # Time domain
    f_mean = np.mean(signal)
    f_std = np.std(signal)
    f_skew = skew(signal)
    f_kurt = kurtosis(signal)
    
    # Frequency domain (PSD using Welch's method)
    freqs, psd = welch(signal, fs, nperseg=fs*2) # 2 second window
    
    # Band powers
    # Delta (0.5-4), Theta (4-8), Alpha (8-13), Beta (13-30), Gamma (30-60)
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 60)
    }
    
    band_powers = []
    for band, (low, high) in bands.items():
        # Find indices corresponding to the frequency band
        idx = np.logical_and(freqs >= low, freqs <= high)
        # Integrate PSD (approximate with sum)
        power = np.sum(psd[idx])
        band_powers.append(power)
        
    features = np.array([f_mean, f_std, f_skew, f_kurt] + band_powers)
    return features

def compute_spectrogram(signal, fs=SAMPLING_RATE, nperseg=64, noverlap=32, target_shape=(64, 64)):
    """
    Computes the spectrogram of the signal and resizes it.
    
    Args:
        signal (np.array): 1D EEG signal.
        fs (float): Sampling rate.
        nperseg (int): Length of each segment for STFT.
        noverlap (int): Overlap between segments.
        target_shape (tuple): Desired output shape (height, width).
        
    Returns:
        np.array: 2D spectrogram (normalized log-magnitude).
    """
    f, t, Sxx = spectrogram(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    # Log-transform to compress dynamic range
    # Add small constant to avoid log(0)
    Sxx_log = np.log(Sxx + 1e-10)
    
    # Normalize to [0, 1] roughly or just standardized
    # Here we do min-max normalization for image-like representation
    S_min = Sxx_log.min()
    S_max = Sxx_log.max()
    if S_max - S_min > 0:
        Sxx_norm = (Sxx_log - S_min) / (S_max - S_min)
    else:
        Sxx_norm = Sxx_log
        
    # Resize to target shape
    # Current shape is (n_freqs, n_time_steps)
    # We want to resize to target_shape
    zoom_factors = (target_shape[0] / Sxx_norm.shape[0], target_shape[1] / Sxx_norm.shape[1])
    Sxx_resized = zoom(Sxx_norm, zoom_factors, order=1) # Bilinear interpolation
    
    return Sxx_resized

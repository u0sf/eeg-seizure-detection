import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config import *
from src.utils import set_seed, plot_confusion_matrix, plot_roc_curve, plot_training_history, plot_signals
from src.data_loader import load_bonn_data, BonnDataset
from src.preprocessing import preprocess_pipeline
from src.features import extract_handcrafted_features, compute_spectrogram
from src.models import Simple1DCNN, Simple2DCNN, ClassicalClassifier
from src.train import train_model, evaluate_model

def main():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Bonn EEG Seizure Detection")
    parser.add_argument("--use-synthetic", action="store_true", help="Use synthetic data if real data is missing")
    args = parser.parse_args()

    # 1. Load Data
    print("\n--- 1. Loading Data ---")
    try:
        X, y = load_bonn_data(allow_synthetic=args.use_synthetic)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
    
    # Binary Classification: Seizure (4) vs Non-Seizure (0,1,2,3)
    # Map 4 -> 1 (Seizure), others -> 0 (Non-Seizure)
    y_binary = np.array([1 if label == 4 else 0 for label in y])
    class_names = ['Non-Seizure', 'Seizure']
    
    print(f"Data Shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y_binary)}")
    
    # Plot raw signals
    plot_signals(X, y_binary, class_names, title="Raw EEG Signals", save_path=f"{ARTIFACTS_DIR}/raw_signals.png")
    
    # 2. Preprocessing & Feature Extraction
    print("\n--- 2. Preprocessing & Feature Extraction ---")
    
    # A. For Classical ML (Handcrafted Features)
    print("Extracting handcrafted features...")
    X_features = []
    for signal in X:
        # Filter first
        sig_filt = preprocess_pipeline(signal, apply_filter=True, apply_norm=False)
        feats = extract_handcrafted_features(sig_filt)
        X_features.append(feats)
    X_features = np.array(X_features)
    print(f"Features Shape: {X_features.shape}")
    
    # B. For 1D CNN (Raw Filtered & Normalized)
    print("Preprocessing for 1D CNN...")
    X_1d = []
    for signal in X:
        # Filter + Normalize
        sig_proc = preprocess_pipeline(signal, apply_filter=True, apply_norm=True)
        X_1d.append(sig_proc)
    X_1d = np.array(X_1d)
    # Add channel dimension for PyTorch: (N, 1, L)
    X_1d = X_1d[:, np.newaxis, :]
    
    # C. For 2D CNN (Spectrograms)
    print("Computing spectrograms for 2D CNN...")
    X_2d = []
    for signal in X:
        # Filter first
        sig_filt = preprocess_pipeline(signal, apply_filter=True, apply_norm=False)
        spec = compute_spectrogram(sig_filt)
        X_2d.append(spec)
    X_2d = np.array(X_2d)
    # Add channel dimension: (N, 1, H, W)
    X_2d = X_2d[:, np.newaxis, :, :]
    print(f"Spectrograms Shape: {X_2d.shape}")
    
    # 3. Splitting Data
    print("\n--- 3. Data Splitting ---")
    # We split indices to keep consistent splits across models
    indices = np.arange(len(y_binary))
    train_idx, test_idx, y_train, y_test = train_test_split(indices, y_binary, test_size=TEST_SIZE, stratify=y_binary, random_state=SEED)
    train_idx, val_idx, y_train, y_val = train_test_split(train_idx, y_train, test_size=VAL_SIZE, stratify=y_train, random_state=SEED)
    
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # --- LEAKAGE CHECK ---
    print("\n--- Checking for Data Leakage ---")
    # 1. Check index intersection
    intersect_train_test = np.intersect1d(train_idx, test_idx)
    intersect_train_val = np.intersect1d(train_idx, val_idx)
    intersect_val_test = np.intersect1d(val_idx, test_idx)
    
    if len(intersect_train_test) > 0 or len(intersect_train_val) > 0 or len(intersect_val_test) > 0:
        print("FAIL: Index overlap detected!")
        print(f"Train-Test Overlap: {len(intersect_train_test)}")
        print(f"Train-Val Overlap: {len(intersect_train_val)}")
        print(f"Val-Test Overlap: {len(intersect_val_test)}")
        sys.exit(1)
    else:
        print("PASS: No index overlap.")

    # 2. Check for duplicate signals (values)
    # This is expensive but good for sanity check on small datasets
    # We check if any test signal is exactly present in training
    # Using a simple hash check or direct comparison
    print("Checking for signal content duplication (Train vs Test)...")
    # Convert to bytes for hashing
    train_hashes = {x.tobytes() for x in X[train_idx]}
    test_hashes = {x.tobytes() for x in X[test_idx]}
    
    overlap_hashes = train_hashes.intersection(test_hashes)
    if len(overlap_hashes) > 0:
        print(f"FAIL: Found {len(overlap_hashes)} duplicate signals between Train and Test!")
        # sys.exit(1) # Optional: strict fail
    else:
        print("PASS: No exact signal duplication found.")
    # ---------------------
    
    # 4. Classical ML Baseline
    print("\n--- 4. Classical ML Baseline (Random Forest) ---")
    clf = ClassicalClassifier('rf')
    clf.train(X_features[train_idx], y_train)
    acc_ml, f1_ml, preds_ml, probs_ml = clf.evaluate(X_features[test_idx], y_test)
    
    # Calculate additional metrics
    prec_ml = precision_score(y_test, preds_ml)
    rec_ml = recall_score(y_test, preds_ml)
    tn, fp, fn, tp = confusion_matrix(y_test, preds_ml).ravel()
    spec_ml = tn / (tn + fp)
    auc_ml = roc_auc_score(y_test, probs_ml)

    print(f"Classical ML - Accuracy: {acc_ml:.4f}, F1: {f1_ml:.4f}")
    print(f"Classical ML - Precision: {prec_ml:.4f}, Recall: {rec_ml:.4f}, Specificity: {spec_ml:.4f}, AUC: {auc_ml:.4f}")
    
    plot_confusion_matrix(y_test, preds_ml, class_names, title="Classical ML Confusion Matrix", save_path=f"{ARTIFACTS_DIR}/cm_ml.png")
    plot_roc_curve(y_test, probs_ml, title="Classical ML ROC Curve", save_path=f"{ARTIFACTS_DIR}/roc_ml.png")
    
    # 5. Deep Learning: 1D CNN
    print("\n--- 5. Deep Learning: 1D CNN ---")
    # Create Datasets
    train_ds_1d = TensorDataset(torch.FloatTensor(X_1d[train_idx]), torch.LongTensor(y_train))
    val_ds_1d = TensorDataset(torch.FloatTensor(X_1d[val_idx]), torch.LongTensor(y_val))
    test_ds_1d = TensorDataset(torch.FloatTensor(X_1d[test_idx]), torch.LongTensor(y_test))
    
    train_loader_1d = DataLoader(train_ds_1d, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_1d = DataLoader(val_ds_1d, batch_size=BATCH_SIZE)
    test_loader_1d = DataLoader(test_ds_1d, batch_size=BATCH_SIZE)
    
    model_1d = Simple1DCNN(num_classes=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_1d.parameters(), lr=LEARNING_RATE)
    
    model_1d, history_1d = train_model(model_1d, train_loader_1d, val_loader_1d, criterion, optimizer, num_epochs=EPOCHS, device=device, model_name='best_model_1d')
    plot_training_history(history_1d, save_path=f"{ARTIFACTS_DIR}/history_1d.png")
    
    y_true_1d, y_pred_1d, y_probs_1d = evaluate_model(model_1d, test_loader_1d, device=device)
    acc_1d = accuracy_score(y_true_1d, y_pred_1d)
    
    # Calculate additional metrics
    prec_1d = precision_score(y_true_1d, y_pred_1d)
    rec_1d = recall_score(y_true_1d, y_pred_1d)
    f1_1d = f1_score(y_true_1d, y_pred_1d)
    tn, fp, fn, tp = confusion_matrix(y_true_1d, y_pred_1d).ravel()
    spec_1d = tn / (tn + fp)
    auc_1d = roc_auc_score(y_true_1d, y_probs_1d[:, 1])

    print(f"1D CNN - Accuracy: {acc_1d:.4f}, F1: {f1_1d:.4f}")
    print(f"1D CNN - Precision: {prec_1d:.4f}, Recall: {rec_1d:.4f}, Specificity: {spec_1d:.4f}, AUC: {auc_1d:.4f}")
    
    plot_confusion_matrix(y_true_1d, y_pred_1d, class_names, title="1D CNN Confusion Matrix", save_path=f"{ARTIFACTS_DIR}/cm_1d.png")
    plot_roc_curve(y_true_1d, y_probs_1d[:, 1], title="1D CNN ROC Curve", save_path=f"{ARTIFACTS_DIR}/roc_1d.png")
    
    # 6. Deep Learning: 2D CNN
    print("\n--- 6. Deep Learning: 2D CNN ---")
    train_ds_2d = TensorDataset(torch.FloatTensor(X_2d[train_idx]), torch.LongTensor(y_train))
    val_ds_2d = TensorDataset(torch.FloatTensor(X_2d[val_idx]), torch.LongTensor(y_val))
    test_ds_2d = TensorDataset(torch.FloatTensor(X_2d[test_idx]), torch.LongTensor(y_test))
    
    train_loader_2d = DataLoader(train_ds_2d, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_2d = DataLoader(val_ds_2d, batch_size=BATCH_SIZE)
    test_loader_2d = DataLoader(test_ds_2d, batch_size=BATCH_SIZE)
    
    model_2d = Simple2DCNN(num_classes=2)
    optimizer = optim.Adam(model_2d.parameters(), lr=LEARNING_RATE)
    
    model_2d, history_2d = train_model(model_2d, train_loader_2d, val_loader_2d, criterion, optimizer, num_epochs=EPOCHS, device=device, model_name='best_model_2d')
    plot_training_history(history_2d, save_path=f"{ARTIFACTS_DIR}/history_2d.png")
    
    y_true_2d, y_pred_2d, y_probs_2d = evaluate_model(model_2d, test_loader_2d, device=device)
    acc_2d = accuracy_score(y_true_2d, y_pred_2d)
    
    # Calculate additional metrics
    prec_2d = precision_score(y_true_2d, y_pred_2d)
    rec_2d = recall_score(y_true_2d, y_pred_2d)
    f1_2d = f1_score(y_true_2d, y_pred_2d)
    tn, fp, fn, tp = confusion_matrix(y_true_2d, y_pred_2d).ravel()
    spec_2d = tn / (tn + fp)
    auc_2d = roc_auc_score(y_true_2d, y_probs_2d[:, 1])

    print(f"2D CNN - Accuracy: {acc_2d:.4f}, F1: {f1_2d:.4f}")
    print(f"2D CNN - Precision: {prec_2d:.4f}, Recall: {rec_2d:.4f}, Specificity: {spec_2d:.4f}, AUC: {auc_2d:.4f}")

    plot_confusion_matrix(y_true_2d, y_pred_2d, class_names, title="2D CNN Confusion Matrix", save_path=f"{ARTIFACTS_DIR}/cm_2d.png")
    plot_roc_curve(y_true_2d, y_probs_2d[:, 1], title="2D CNN ROC Curve", save_path=f"{ARTIFACTS_DIR}/roc_2d.png")
    
    # 7. Comparison
    print("\n--- 7. Final Comparison ---")
    print(f"{'Model':<20} | {'Accuracy':<10}")
    print("-" * 35)
    print(f"{'Classical ML':<20} | {acc_ml:.4f}")
    print(f"{'1D CNN':<20} | {acc_1d:.4f}")
    print(f"{'2D CNN':<20} | {acc_2d:.4f}")

if __name__ == "__main__":
    main()

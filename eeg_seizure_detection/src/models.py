import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

class Simple1DCNN(nn.Module):
    """
    A simple 1D CNN for raw EEG signals.
    Input: (Batch, 1, 4097)
    """
    def __init__(self, num_classes=2):
        super(Simple1DCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=10, stride=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # Block 2
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=10, stride=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # Block 3
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=10, stride=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # Fully Connected
        # We need to calculate the size after pooling. 
        # Input 4097
        # After Block 1: (4097 - 10 + 1) / 4 = 1022
        # After Block 2: (1022 - 10 + 1) / 4 = 253
        # After Block 3: (253 - 10 + 1) / 4 = 61
        self.fc1 = nn.Linear(64 * 61, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Ensure input is (Batch, 1, Length)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class Simple2DCNN(nn.Module):
    """
    A simple 2D CNN for spectrogram images.
    Input: (Batch, 1, 64, 64)
    """
    def __init__(self, num_classes=2):
        super(Simple2DCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2) # 64 -> 32
        
        # Block 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2) # 32 -> 16
        
        # Block 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2) # 16 -> 8
        
        # FC
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Ensure input is (Batch, 1, H, W)
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ClassicalClassifier:
    """
    Wrapper for Scikit-Learn classifiers.
    """
    def __init__(self, model_type='rf'):
        if model_type == 'rf':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'svm':
            self.model = SVC(probability=True, random_state=42)
        else:
            raise ValueError("Unknown model type")
            
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_probs = self.model.predict_proba(X_test)[:, 1] # Probability of positive class
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        return acc, f1, y_pred, y_probs

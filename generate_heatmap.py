import os
import joblib
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# --- Configuration ---
DATA_DIR = r"C:\Users\bhavy\Documents\Project\SecureShelf\ProcessedData"
INPUT_FEATURES = 378
MODEL_PATH = os.path.join(DATA_DIR, "best_v4.pth")
SCALER_PATH = os.path.join(DATA_DIR, "scaler_v4.pkl")
OUTPUT_PATH = r"C:\Users\bhavy\Documents\Project\SecureShelf\confusion_matrix_v4.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Definitions (from evaluate_v4.py) ---
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return torch.sum(weights * x, dim=1)

class PoseGRU_V4(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 64
        self.num_layers = 1
        self.conv = nn.Conv1d(in_channels=INPUT_FEATURES, out_channels=128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(128, self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.attn = Attention(self.hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(self.hidden_size, 2)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.conv(x))
        x = x.transpose(1, 2)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.gru(x, h0)
        out = self.layer_norm(self.attn(out))
        out = self.dropout(out)
        return self.fc(out)

def main():
    print("Loading data and model...")
    # 1. Load Data
    X = np.load(os.path.join(DATA_DIR, "X_features_v4.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_labels_v4.npy"))

    # 2. Split (Match evaluate_v4.py exactly)
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    # 3. Scale
    scaler = joblib.load(SCALER_PATH)
    samples_test, frames, features = X_test.shape
    X_test_scaled = scaler.transform(X_test.reshape(-1, features)).reshape(samples_test, frames, features)

    # 4. Load Model
    model = PoseGRU_V4().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # 5. Predict
    print("Running evaluation...")
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test_scaled).to(device)
        logits = model(X_tensor)
        y_pred = torch.argmax(logits, dim=1).cpu().numpy()

    # 6. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # 7. Plotting
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="white")
    
    # Use professional color palette (Blues)
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Normal', 'Shoplifting'], 
                yticklabels=['Normal', 'Shoplifting'])
    
    plt.title('SecureShelf Model: Confusion Matrix (V4 Heatmap)', fontsize=14, pad=20)
    plt.xlabel('Predicted Action', fontsize=12, labelpad=10)
    plt.ylabel('Actual Action', fontsize=12, labelpad=10)
    
    # Save image
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300)
    print(f"Heatmap saved successfully to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

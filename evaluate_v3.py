import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

DATA_DIR = r"C:\Users\bhavy\Documents\Project\SecureShelf\ProcessedData"
INPUT_FEATURES = 378
MODEL_PATH = os.path.join(DATA_DIR, "best_v3.pth")
SCALER_PATH = os.path.join(DATA_DIR, "scaler_v3.pkl")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return torch.sum(weights * x, dim=1)

class PoseBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 64
        self.num_layers = 1

        self.conv = nn.Conv1d(in_channels=INPUT_FEATURES, out_channels=128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(
            128,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
        )

        lstm_out_dim = self.hidden_size * 2
        self.attn = Attention(lstm_out_dim)
        self.layer_norm = nn.LayerNorm(lstm_out_dim)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(lstm_out_dim, 2)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.conv(x))
        x = x.transpose(1, 2)

        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))

        out = self.layer_norm(self.attn(out))
        out = self.dropout(out)
        return self.fc(out)

def main():
    X = np.load(os.path.join(DATA_DIR, "X_features_v3.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_labels_v3.npy"))

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

    scaler = joblib.load(SCALER_PATH)

    samples_test, frames, features = X_test.shape
    X_test_scaled = scaler.transform(X_test.reshape(-1, features)).reshape(samples_test, frames, features)

    model = PoseBiLSTM().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test_scaled).to(device)
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        y_pred = np.argmax(probs, axis=1)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1], zero_division=0)

    print("=== V3 BiLSTM Evaluation (Held-out Test Split) ===")
    print(f"Test samples: {len(y_test)}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nConfusion Matrix (rows=true, cols=pred)")
    print("           Pred Normal  Pred Shoplifting")
    print(f"True Normal     {cm[0, 0]:>3}           {cm[0, 1]:>3}")
    print(f"True Shoplift   {cm[1, 0]:>3}           {cm[1, 1]:>3}")

    print("\nPer-class metrics:")
    print(f"Normal      | Precision: {precision[0]:.3f}  Recall: {recall[0]:.3f}  F1: {f1[0]:.3f}")
    print(f"Shoplifting | Precision: {precision[1]:.3f}  Recall: {recall[1]:.3f}  F1: {f1[1]:.3f}")

    report_path = os.path.join(DATA_DIR, "evaluation_v3_confusion_matrix.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== V3 BiLSTM Evaluation (Held-out Test Split) ===\n")
        f.write(f"Test samples: {len(y_test)}\n")
        f.write(f"Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write("Confusion Matrix (rows=true, cols=pred)\n")
        f.write("           Pred Normal  Pred Shoplifting\n")
        f.write(f"True Normal     {cm[0, 0]:>3}           {cm[0, 1]:>3}\n")
        f.write(f"True Shoplift   {cm[1, 0]:>3}           {cm[1, 1]:>3}\n\n")
        f.write("Per-class metrics:\n")
        f.write(f"Normal      | Precision: {precision[0]:.3f}  Recall: {recall[0]:.3f}  F1: {f1[0]:.3f}\n")
        f.write(f"Shoplifting | Precision: {precision[1]:.3f}  Recall: {recall[1]:.3f}  F1: {f1[1]:.3f}\n")

if __name__ == "__main__":
    main()
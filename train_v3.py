import os
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# ================= CONFIG =================
DATA_DIR = r"C:\Users\bhavy\Documents\Project\SecureShelf\ProcessedData"
INPUT_FEATURES = 378
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def augment_dataset(X, y):
    X_aug, y_aug = [], []
    for i in range(len(X)):
        seq = X[i]
        label = y[i]
        X_aug.append(seq)
        y_aug.append(label)

        for noise_scale in [0.005, 0.01, 0.015, 0.02, 0.03]:
            X_aug.append(seq + np.random.normal(0, noise_scale, seq.shape))
            y_aug.append(label)

        mask = np.random.binomial(1, 0.8, seq.shape)
        X_aug.append(seq * mask)
        y_aug.append(label)

        fast = seq[::2]
        if len(fast) < seq.shape[0]:
            pad = np.zeros_like(seq)
            pad[: len(fast)] = fast
            X_aug.append(pad)
            y_aug.append(label)

        slow = np.repeat(seq, 2, axis=0)[: seq.shape[0]]
        X_aug.append(slow)
        y_aug.append(label)

    return np.array(X_aug), np.array(y_aug)

class ActionDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.tensor(self.labels[idx])

# ================= BiLSTM MODEL =================
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

        # Bidirectional LSTM mapping
        self.lstm = nn.LSTM(
            128,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Because bidirectional=True, output dimension is doubled
        lstm_out_dim = self.hidden_size * 2

        self.attn = Attention(lstm_out_dim)
        self.layer_norm = nn.LayerNorm(lstm_out_dim)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(lstm_out_dim, 2)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.conv(x))
        x = x.transpose(1, 2)

        # LSTM requires h0 and c0
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        
        out, _ = self.lstm(x, (h0, c0))

        out = self.layer_norm(self.attn(out))
        out = self.dropout(out)
        return self.fc(out)

if __name__ == "__main__":
    X = np.load(os.path.join(DATA_DIR, "X_features_v3.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_labels_v3.npy"))

    print("Target distribution:", np.bincount(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    X_train, y_train = augment_dataset(X_train, y_train)

    samples_train, frames, features = X_train.shape
    samples_test = X_test.shape[0]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, features)).reshape(samples_train, frames, features)
    X_test_scaled = scaler.transform(X_test.reshape(-1, features)).reshape(samples_test, frames, features)

    joblib.dump(scaler, os.path.join(DATA_DIR, "scaler_v3.pkl"))

    train_loader = DataLoader(ActionDataset(X_train_scaled, y_train), batch_size=32, shuffle=True)
    test_loader = DataLoader(ActionDataset(X_test_scaled, y_test), batch_size=32, shuffle=False)

    model = PoseBiLSTM().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    best_acc = 0
    patience = 50
    patience_counter = 0

    print("\nStarting Clean V3 (BiLSTM) Training...")

    for epoch in range(150):
        model.train()
        train_loss = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                out = model(x_batch)
                _, pred = torch.max(out, 1)
                total += y_batch.size(0)
                correct += (pred == y_batch).sum().item()

        acc = 100 * correct / total

        if epoch % 5 == 0:
            print(f"Epoch {epoch:03d} | Loss: {train_loss / len(train_loader):.4f} | Val Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(DATA_DIR, "best_v3.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stop at epoch {epoch}. Best Val Acc: {best_acc:.2f}%")
                break

    print(f"\nTraining Complete. Validated Best Accuracy (BiLSTM): {best_acc:.2f}%")
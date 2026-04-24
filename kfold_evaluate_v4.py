import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

# ================= CONFIG =================
DATA_DIR = r"C:\Users\bhavy\Documents\Project\SecureShelf\ProcessedData"
INPUT_FEATURES = 378
K_FOLDS = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def augment_dataset(X, y):
    X_aug, y_aug = [], []
    for i in range(len(X)):
        seq = X[i]
        label = y[i]
        X_aug.append(seq)
        y_aug.append(label)

        # Jitter
        for noise_scale in [0.005, 0.01, 0.015, 0.02, 0.03]:
            X_aug.append(seq + np.random.normal(0, noise_scale, seq.shape))
            y_aug.append(label)

        # Masking
        mask = np.random.binomial(1, 0.8, seq.shape)
        X_aug.append(seq * mask)
        y_aug.append(label)

        # Fast
        fast = seq[::2]
        if len(fast) < seq.shape[0]:
            pad = np.zeros_like(seq)
            pad[: len(fast)] = fast
            X_aug.append(pad)
            y_aug.append(label)

        # Slow
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

# ================= GRU MODEL =================
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return torch.sum(weights * x, dim=1)

class PoseGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 64
        self.num_layers = 1

        self.conv = nn.Conv1d(
            in_channels=INPUT_FEATURES,
            out_channels=128,
            kernel_size=3,
            padding=1,
        )
        self.relu = nn.ReLU()

        self.gru = nn.GRU(
            128,
            self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=False,
        )

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

if __name__ == "__main__":
    # Load the 40-frame V4 data
    X = np.load(os.path.join(DATA_DIR, "X_features_v4.npy"))
    y = np.load(os.path.join(DATA_DIR, "y_labels_v4.npy"))

    print(f"Total Dataset Size: {len(y)} sequences")
    print(f"Target distribution: {np.bincount(y)}")
    print(f"Starting {K_FOLDS}-Fold Stratified Cross Validation (V4 GRU)...\n")

    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
    
    fold_metrics = {'acc': [], 'prec': [], 'rec': [], 'f1': []}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"--- FOLD {fold}/{K_FOLDS} ---")
        
        # 1. Split Data
        X_train_raw, X_test = X[train_idx], X[test_idx]
        y_train_raw, y_test = y[train_idx], y[test_idx]

        # 2. Augment ONLY the training fold
        X_train_aug, y_train_aug = augment_dataset(X_train_raw, y_train_raw)

        # 3. Scale features (Fit only on training fold)
        samples_train, frames, features = X_train_aug.shape
        samples_test = X_test.shape[0]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_aug.reshape(-1, features)).reshape(samples_train, frames, features)
        X_test_scaled = scaler.transform(X_test.reshape(-1, features)).reshape(samples_test, frames, features)

        # 4. DataLoaders
        train_loader = DataLoader(ActionDataset(X_train_scaled, y_train_aug), batch_size=32, shuffle=True)
        test_loader = DataLoader(ActionDataset(X_test_scaled, y_test), batch_size=32, shuffle=False)

        # 5. Initialize a FRESH GRU model for this fold
        model = PoseGRU().to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

        best_val_acc = 0
        patience, patience_counter = 40, 0
        best_model_state = None

        # 6. Train the fold
        for epoch in range(100): 
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x_batch), y_batch)
                loss.backward()
                optimizer.step()
            
            scheduler.step()

            # Evaluate fold validation
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                    _, pred = torch.max(model(x_batch), 1)
                    total += y_batch.size(0)
                    correct += (pred == y_batch).sum().item()

            acc = correct / total

            if acc > best_val_acc:
                best_val_acc = acc
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # 7. Final Evaluation for this Fold using the best weights
        model.load_state_dict(best_model_state)
        model.eval()
        
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                logits = model(x_batch)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                all_preds.extend(preds)
                all_targets.extend(y_batch.numpy())

        fold_acc = accuracy_score(all_targets, all_preds)
        fold_prec, fold_rec, fold_f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro', zero_division=0)

        fold_metrics['acc'].append(fold_acc)
        fold_metrics['prec'].append(fold_prec)
        fold_metrics['rec'].append(fold_rec)
        fold_metrics['f1'].append(fold_f1)

        print(f"Fold {fold} Results -> Acc: {fold_acc*100:.2f}% | Prec: {fold_prec:.3f} | Rec: {fold_rec:.3f} | F1: {fold_f1:.3f}\n")

    # ================= FINAL AGGREGATE =================
    print("="*40)
    print("FINAL K-FOLD CROSS-VALIDATION RESULTS (V4 GRU)")
    print("="*40)
    print(f"Average Accuracy  : {np.mean(fold_metrics['acc'])*100:.2f}% (+/- {np.std(fold_metrics['acc'])*100:.2f}%)")
    print(f"Average Precision : {np.mean(fold_metrics['prec']):.3f}")
    print(f"Average Recall    : {np.mean(fold_metrics['rec']):.3f}")
    print(f"Average F1-Score  : {np.mean(fold_metrics['f1']):.3f}")
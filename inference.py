import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import joblib
import threading
from ultralytics import YOLO

# --- Configuration ---
ROOT_DIR = r"C:\Users\bhavy\Documents\Project\SecureShelf"
PROCESSED_DIR = os.path.join(ROOT_DIR, "ProcessedData")
YOLO_MODEL_PATH = os.path.join(ROOT_DIR, "yolo26n-pose.pt")
GRU_MODEL_PATH = os.path.join(PROCESSED_DIR, "best_v4.pth")
SCALER_PATH = os.path.join(PROCESSED_DIR, "scaler_v4.pkl")

MAX_FRAMES = 40
MAX_PEOPLE = 3
INPUT_FEATURES = 378
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Definitions ---
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
        self.conv = nn.Conv1d(in_channels=189, out_channels=128, kernel_size=3, padding=1) # 189 is base features
        self.relu = nn.ReLU()
        self.gru = nn.GRU(128, self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.attn = Attention(self.hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(self.hidden_size, 2)

    def forward(self, x):
        # x is (batch, frames, features_total=378)
        # We need to split into Base and Velocity or just take the first half if needed?
        # Actually train_v4 used 378 input features.
        # Wait, let's re-check train_v4.py line 85: in_channels=INPUT_FEATURES (378)
        pass

# Corrected model based on train_v4.py
class PoseGRU_V4(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 64
        self.num_layers = 1
        self.conv = nn.Conv1d(in_channels=378, out_channels=128, kernel_size=3, padding=1)
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
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(DEVICE)
        out, _ = self.gru(x, h0)
        out = self.layer_norm(self.attn(out))
        out = self.dropout(out)
        return self.fc(out)

def get_angle(p1, p2, p3):
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]])
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return 0.0
    cos_theta = np.dot(v1, v2) / (n1 * n2)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

def get_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

_YOLO_MODEL = None
_GRU_MODEL = None
_SCALER = None
_MODEL_LOCK = threading.Lock()

def get_yolo():
    global _YOLO_MODEL
    if _YOLO_MODEL is None:
        _YOLO_MODEL = YOLO(YOLO_MODEL_PATH)
    return _YOLO_MODEL

def get_gru():
    global _GRU_MODEL
    if _GRU_MODEL is None:
        _GRU_MODEL = PoseGRU_V4().to(DEVICE)
        _GRU_MODEL.load_state_dict(torch.load(GRU_MODEL_PATH, map_location=DEVICE))
        _GRU_MODEL.eval()
    return _GRU_MODEL

def get_scaler():
    global _SCALER
    if _SCALER is None:
        _SCALER = joblib.load(SCALER_PATH)
    return _SCALER

class SecureShelfEngine:
    def __init__(self):
        self.yolo = get_yolo()
        self.gru = get_gru()
        self.scaler = get_scaler()
        self.feature_buffer = []

    def process_frame(self, frame):
        # 1. Run YOLO Tracking (Shared instance with Lock)
        with _MODEL_LOCK:
            results = self.yolo.track(frame, persist=True, verbose=False)
        frame_data = np.zeros((MAX_PEOPLE, 63))
        
        if len(results[0].boxes) > 0 and results[0].boxes.id is not None:
            ids = results[0].boxes.id.int().cpu().tolist()
            keypoints = results[0].keypoints.data.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            
            next_slot = 0
            for i, track_id in enumerate(ids):
                if next_slot < MAX_PEOPLE:
                    slot = next_slot
                    next_slot += 1
                    
                    kp = keypoints[i]
                    box = boxes[i]
                    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
                    diag = np.sqrt((box[2] - box[0])**2 + (box[3] - box[1])**2) + 1e-6
                    
                    norm_kp = np.zeros((17, 3))
                    for j in range(17):
                        if kp[j][2] > 0.1:
                            norm_kp[j][0] = (kp[j][0] - cx) / diag
                            norm_kp[j][1] = (kp[j][1] - cy) / diag
                            norm_kp[j][2] = kp[j][2]
                    
                    frame_data[slot][:51] = norm_kp.flatten()
                    
                    if len(kp) >= 17:
                        angles = [
                            get_angle(kp[5], kp[7], kp[9]), get_angle(kp[6], kp[8], kp[10]),
                            get_angle(kp[11], kp[13], kp[15]), get_angle(kp[12], kp[14], kp[16]),
                            get_angle(kp[7], kp[5], kp[11]), get_angle(kp[8], kp[6], kp[12]),
                            get_angle(kp[5], kp[11], kp[13]), get_angle(kp[6], kp[12], kp[14])
                        ]
                        distances = [
                            get_dist(norm_kp[9][:2], norm_kp[11][:2]), get_dist(norm_kp[10][:2], norm_kp[12][:2]),
                            get_dist(norm_kp[9][:2], norm_kp[10][:2]), get_dist(norm_kp[5][:2], norm_kp[9][:2])
                        ]
                        frame_data[slot][51:59] = angles
                        frame_data[slot][59:63] = distances
        
        current_raw = frame_data.flatten()
        self.feature_buffer.append(current_raw)
        
        if len(self.feature_buffer) > MAX_FRAMES:
            self.feature_buffer.pop(0)
            
        if len(self.feature_buffer) == MAX_FRAMES:
            seq = np.array(self.feature_buffer)
            velocity = np.zeros_like(seq)
            velocity[1:] = seq[1:] - seq[:-1]
            full_seq = np.concatenate([seq, velocity], axis=1)
            scaled_seq = self.scaler.transform(full_seq).reshape(1, MAX_FRAMES, INPUT_FEATURES)
            
            with _MODEL_LOCK:
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(scaled_seq).to(DEVICE)
                    output = self.gru(input_tensor)
                    probs = torch.softmax(output, dim=1)
                    conf, pred = torch.max(probs, 1)
                
            return {
                "prediction": "Shoplifting" if pred.item() == 1 else "Normal",
                "confidence": conf.item(),
                "probs": probs.cpu().numpy().tolist()[0],
                "results": results
            }
            
        return {"prediction": "Initializing...", "confidence": 0, "results": results}

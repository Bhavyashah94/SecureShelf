import os
from glob import glob
import cv2
import random
import time
from datetime import datetime, timezone
from collections import deque
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import numpy as np
import torch
import torch.nn as nn
import joblib

app = Flask(__name__)

# ================= CONFIG =================
DATA_DIR = r"C:\Users\bhavy\Documents\Project\SecureShelf\ProcessedData"
MODEL_PATH = os.path.join(DATA_DIR, 'best_v4.pth')
SCALER_PATH = os.path.join(DATA_DIR, 'scaler_v4.pkl')
DATASET_DIR = r"C:\Users\bhavy\Documents\Project\SecureShelf\Dataset"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_FEATURES = 378
MAX_FRAMES = 40
MAX_PEOPLE = 3
ALERT_THRESHOLD = 50
DEFAULT_CAMERA_FPS = 25.0
ALERT_HOLD_MULTIPLIER = 2.0
MIN_ALERT_HOLD_SECONDS = 6.0

# ================= VIDEO SETUP =================
CAMERA_CONFIG = [
    {"id": 1, "name": "CAM-01", "location": "Aisle 1", "pool": "Normal"},
    {"id": 2, "name": "CAM-02", "location": "Electronics", "pool": "Shoplifting"},
]

def discover_videos(folder_name):
    pattern = os.path.join(DATASET_DIR, folder_name, "*.mp4")
    return sorted(glob(pattern))

SOURCE_POOLS = {
    camera["id"]: discover_videos(camera["pool"])
    for camera in CAMERA_CONFIG
}

VIDEO_POOLS = {
    camera["id"]: []
    for camera in CAMERA_CONFIG
}

def resolve_pose_model_path():
    preferred_models = [
        r"C:\Users\bhavy\Documents\Project\SecureShelf\yolo26n-pose.pt",
        os.path.join(os.getcwd(), "yolo26n-pose.pt"),
    ]

    for model_path in preferred_models:
        if os.path.exists(model_path):
            return model_path

    return "yolo26n-pose.pt"

POSE_MODEL_PATH = resolve_pose_model_path()
print(f"Loading pose model: {os.path.basename(POSE_MODEL_PATH)}")
yolo_model = YOLO(POSE_MODEL_PATH)
DETECTION_STRIDE = 2

# ================= LSTM CLASSIFIER =================
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
        self.conv = nn.Conv1d(INPUT_FEATURES, 128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.gru = nn.GRU(128, self.hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=False)
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

try:
    print("Loading LSTM classifier...")
    lstm_model = PoseGRU().to(DEVICE)
    lstm_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    lstm_model.eval()
    scaler = joblib.load(SCALER_PATH)
    print("LSTM classifier loaded successfully")
except Exception as e:
    print(f"Warning: Could not load LSTM model: {e}")
    lstm_model = None
    scaler = None

# ================= GLOBAL STATE =================
camera_status = {camera["id"]: "PLAYBACK" for camera in CAMERA_CONFIG}
camera_confidence = {camera["id"]: 0 for camera in CAMERA_CONFIG}

camera_runtime = {
    camera["id"]: {
        "capture": None,
        "playlist": deque(),
        "current_video": None,
        "last_annotated_frame": None,
        "last_detection_count": 0,
        "last_detection_confidence": 0,
        "frame_count": 0,
        "pose_sequence": [],
        "theft_probability": 0,
        "alert_hold_until": 0.0,
        "clip_fps": DEFAULT_CAMERA_FPS,
        "delay_frames": MAX_FRAMES * max(1, DETECTION_STRIDE),
        "output_buffer": deque(),
    }
    for camera in CAMERA_CONFIG
}

# ================= HELPERS =================

def reset_runtime_state(camera_id):
    state = camera_runtime[camera_id]
    state["frame_count"] = 0
    state["pose_sequence"] = []
    state["theft_probability"] = 0
    state["alert_hold_until"] = 0.0
    state["clip_fps"] = DEFAULT_CAMERA_FPS
    state["delay_frames"] = MAX_FRAMES * max(1, DETECTION_STRIDE)
    state["output_buffer"].clear()
    state["last_annotated_frame"] = None
    state["last_detection_count"] = 0
    state["last_detection_confidence"] = 0
    camera_status[camera_id] = "PLAYBACK"
    camera_confidence[camera_id] = 0

def get_sequence_window_seconds(camera_id):
    state = camera_runtime[camera_id]
    clip_fps = state.get("clip_fps", DEFAULT_CAMERA_FPS)
    if clip_fps is None or clip_fps <= 0:
        clip_fps = DEFAULT_CAMERA_FPS

    # We append one feature vector every DETECTION_STRIDE source frames.
    effective_feature_fps = clip_fps / max(1, DETECTION_STRIDE)
    if effective_feature_fps <= 0:
        effective_feature_fps = DEFAULT_CAMERA_FPS / max(1, DETECTION_STRIDE)

    return MAX_FRAMES / effective_feature_fps

def get_angle(p1, p2, p3):
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]])
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos_theta = np.dot(v1, v2) / (n1 * n2)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

def get_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def extract_frame_features(frame, yolo_results):
    """Extract pose features from a single frame for LSTM input."""
    frame_data = np.zeros((MAX_PEOPLE, 63))
    
    if not yolo_results or not yolo_results[0].boxes or len(yolo_results[0].boxes) == 0:
        return frame_data.flatten()
    
    results = yolo_results[0]
    keypoints = results.keypoints.data.cpu().numpy() if hasattr(results, 'keypoints') else None
    boxes = results.boxes.xyxy.cpu().numpy() if hasattr(results.boxes, 'xyxy') else None
    
    if keypoints is None or boxes is None:
        return frame_data.flatten()
    
    for slot in range(min(len(keypoints), MAX_PEOPLE)):
        kp = keypoints[slot]
        box = boxes[slot]
        
        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        diag = np.sqrt((box[2] - box[0])**2 + (box[3] - box[1])**2) + 1e-6
        
        norm_kp = np.zeros((17, 3))
        for j in range(min(17, len(kp))):
            if kp[j][2] > 0.1:
                norm_kp[j][0] = (kp[j][0] - cx) / diag
                norm_kp[j][1] = (kp[j][1] - cy) / diag
                norm_kp[j][2] = kp[j][2]
        
        frame_data[slot][:51] = norm_kp.flatten()
        
        if len(kp) >= 17:
            angles = [
                get_angle(kp[5], kp[7], kp[9]),
                get_angle(kp[6], kp[8], kp[10]),
                get_angle(kp[11], kp[13], kp[15]),
                get_angle(kp[12], kp[14], kp[16]),
                get_angle(kp[7], kp[5], kp[11]),
                get_angle(kp[8], kp[6], kp[12]),
                get_angle(kp[5], kp[11], kp[13]),
                get_angle(kp[6], kp[12], kp[14]),
            ]
            distances = [
                get_dist(norm_kp[9][:2], norm_kp[11][:2]),
                get_dist(norm_kp[10][:2], norm_kp[12][:2]),
                get_dist(norm_kp[9][:2], norm_kp[10][:2]),
                get_dist(norm_kp[5][:2], norm_kp[9][:2]),
            ]
            frame_data[slot][51:59] = angles
            frame_data[slot][59:63] = distances
    
    return frame_data.flatten()

def refill_playlist(camera_id):
    playlist = VIDEO_POOLS.get(camera_id, [])[:]
    random.shuffle(playlist)
    camera_runtime[camera_id]["playlist"] = deque(playlist)

def open_next_clip(camera_id):
    state = camera_runtime[camera_id]

    while True:
        if not state["playlist"]:
            refill_playlist(camera_id)

        if not state["playlist"]:
            return None

        next_video = state["playlist"].popleft()
        capture = cv2.VideoCapture(next_video)

        if capture.isOpened():
            reset_runtime_state(camera_id)
            state["capture"] = capture
            state["current_video"] = next_video
            clip_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            state["clip_fps"] = clip_fps if clip_fps > 0 else DEFAULT_CAMERA_FPS
            state["delay_frames"] = max(
                1,
                int(round(get_sequence_window_seconds(camera_id) * state["clip_fps"])),
            )
            return capture

        capture.release()

def get_capture_frame(camera_id):
    state = camera_runtime[camera_id]

    if state["capture"] is None or not state["capture"].isOpened():
        if open_next_clip(camera_id) is None:
            return None, None

    ret, frame = state["capture"].read()
    if ret:
        return frame, state["current_video"]

    state["capture"].release()
    state["capture"] = None
    return get_capture_frame(camera_id)


def build_status_summary():
    active_cameras = len(camera_status)
    source_clips = sum(len(videos) for videos in SOURCE_POOLS.values())
    detection_cameras = sum(1 for status in camera_status.values() if status in ["POSE", "ALERT"])
    threat_cameras = sum(1 for status in camera_status.values() if status == "ALERT")
    average_confidence = 0
    if camera_confidence:
        average_confidence = int(sum(camera_confidence.values()) / len(camera_confidence))

    return {
        "status": camera_status,
        "confidence": camera_confidence,
        "current_video": {
            camera_id: os.path.basename(state["current_video"]) if state["current_video"] else None
            for camera_id, state in camera_runtime.items()
        },
        "active_cameras": active_cameras,
        "source_clips": source_clips,
        "alert_cameras": threat_cameras,
        "detection_cameras": detection_cameras,
        "average_confidence": average_confidence,
        "pipeline_mode": "direct_playback",
        "sequence_window_seconds": {
            camera_id: round(get_sequence_window_seconds(camera_id), 2)
            for camera_id in camera_runtime
        },
        "stream_delay_frames": {
            camera_id: camera_runtime[camera_id]["delay_frames"]
            for camera_id in camera_runtime
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

# ================= MAIN PIPELINE =================

def classify_pose_sequence(camera_id):
    """Run LSTM on accumulated pose frames to classify theft."""
    state = camera_runtime[camera_id]
    
    if lstm_model is None or scaler is None:
        return 0
    
    if len(state["pose_sequence"]) < MAX_FRAMES:
        return 0
    
    try:
        X_batch_list = state["pose_sequence"][-MAX_FRAMES:]
        X_batch = np.array(X_batch_list)
        
        if X_batch.shape[0] != MAX_FRAMES:
            return 0
        
        X_batch_reshaped = X_batch.reshape(MAX_FRAMES, MAX_PEOPLE, 63)
        
        raw_features = X_batch_reshaped
        velocity = np.zeros_like(raw_features)
        velocity[1:] = raw_features[1:] - raw_features[:-1]
        final_sequence = np.concatenate([raw_features, velocity], axis=2)
        
        # Flatten people dimension: (40, 3, 126) -> (40, 378)
        final_sequence_flat = final_sequence.reshape(MAX_FRAMES, MAX_PEOPLE * 126)
        
        # Scale features
        X_scaled = scaler.transform(final_sequence_flat)
        
        # Create tensor for model: (1, 40, 378)
        X_tensor = torch.FloatTensor(X_scaled).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = lstm_model(X_tensor)
            probs = torch.softmax(logits, dim=1)
            theft_prob = float(probs[0, 1].item()) * 100
            state["theft_probability"] = int(theft_prob)
            print(f"[DEBUG] Camera {camera_id}: LSTM inference - theft_prob={theft_prob:.2f}%, class_0={probs[0, 0].item()*100:.2f}%, class_1={probs[0, 1].item()*100:.2f}%")
            
            return int(theft_prob)
    except Exception as e:
        print(f"LSTM inference error on camera {camera_id}: {e}")
        import traceback
        traceback.print_exc()
        return 0


def annotate_frame_with_pose(frame, camera_id, current_video):
    state = camera_runtime[camera_id]
    results = yolo_model.predict(frame, verbose=False)

    annotated = frame.copy()
    detection_count = 0
    confidence_score = 0

    if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
        detection_count = len(results[0].boxes)
        if hasattr(results[0].boxes, "conf") and results[0].boxes.conf is not None:
            try:
                confidence_score = int(float(results[0].boxes.conf.mean().item()) * 100)
            except:
                confidence_score = 0
        
        camera_confidence[camera_id] = confidence_score
        
        try:
            annotated = results[0].plot()
            if annotated is None:
                annotated = frame.copy()
        except Exception as e:
            print(f"Pose overlay error for camera {camera_id}: {e}")
            annotated = frame.copy()
    
    # Extract features and accumulate for LSTM
    frame_features = extract_frame_features(frame, results)
    state["pose_sequence"].append(frame_features)
    if len(state["pose_sequence"]) > MAX_FRAMES * 2:
        state["pose_sequence"] = state["pose_sequence"][-MAX_FRAMES:]
    
    # Keep last theft probability between classification frames.
    theft_prob = state["theft_probability"]
    if state["frame_count"] % 10 == 0:
        print(f"[DEBUG] Camera {camera_id}: Calling LSTM classification at frame {state['frame_count']}, sequence length: {len(state['pose_sequence'])}")
        theft_prob = classify_pose_sequence(camera_id)

    now = time.monotonic()
    if theft_prob > ALERT_THRESHOLD:
        hold_seconds = max(
            MIN_ALERT_HOLD_SECONDS,
            get_sequence_window_seconds(camera_id) * ALERT_HOLD_MULTIPLIER,
        )
        state["alert_hold_until"] = now + hold_seconds

    alert_active = now < state["alert_hold_until"]
    
    # Update status based on detection
    if alert_active:
        camera_status[camera_id] = "ALERT"
    elif detection_count > 0:
            camera_status[camera_id] = "POSE"
    else:
        camera_status[camera_id] = "NO POSE"
        camera_confidence[camera_id] = 0
    
    state["last_detection_count"] = detection_count
    state["last_detection_confidence"] = confidence_score
    state["last_annotated_frame"] = annotated

    clip_label = f"CLIP: {os.path.basename(current_video) if current_video else 'Unknown'}"
    
    # Build overlay with threat info
    if alert_active:
        threat_label = f"SHOPLIFTING ALERT: {state['theft_probability']}%"
        text_color = (0, 0, 255)
        box_color = (0, 0, 139)
    else:
        threat_label = f"POSE: {detection_count}" if detection_count else "NO POSE"
        text_color = (0, 255, 0)
        box_color = (0, 100, 0)
    
    overlay_label = f"{clip_label} | {threat_label}"
    cv2.rectangle(annotated, (12, 12), (640, 70), box_color, -1)
    cv2.putText(annotated, overlay_label, (24, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
    
    # Add alert border if threat
    if alert_active:
        cv2.rectangle(annotated, (0, 0), (640, 480), (0, 0, 255), 3)
    
    return annotated


def generate_frames(camera_id):
    if not VIDEO_POOLS.get(camera_id):
        raise RuntimeError(f"No videos found for camera {camera_id}")

    while True:
        frame, current_video = get_capture_frame(camera_id)
        if frame is None:
            continue

        frame = cv2.resize(frame, (640, 480))
        state = camera_runtime[camera_id]
        state["frame_count"] += 1

        should_detect = (
            state["last_annotated_frame"] is None
            or state["frame_count"] % DETECTION_STRIDE == 0
        )

        if should_detect:
            frame = annotate_frame_with_pose(frame, camera_id, current_video)
        else:
            cached_frame = state["last_annotated_frame"]
            if cached_frame is not None:
                frame = cached_frame.copy()

        # Delay output by one full LSTM sequence window so displayed frames have processed context.
        state["output_buffer"].append(frame.copy())
        if len(state["output_buffer"]) <= state["delay_frames"]:
            progress = int((len(state["output_buffer"]) / max(1, state["delay_frames"])) * 100)
            loading_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(
                loading_frame,
                f"CAM-{camera_id:02d} BUFFERING... {progress}%",
                (70, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                loading_frame,
                "Waiting for full LSTM sequence window",
                (85, 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            _, buffer = cv2.imencode('.jpg', loading_frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            continue

        delayed_frame = state["output_buffer"].popleft()
        _, buffer = cv2.imencode('.jpg', delayed_frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


def cleanup_runtime():
    for state in camera_runtime.values():
        capture = state.get("capture")
        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass
            state["capture"] = None

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

# ================= ROUTES =================
@app.route('/')
def index():
    return render_template('index.html', cameras=CAMERA_CONFIG)

@app.route('/video_feed/<int:camera_id>')
def video_feed(camera_id):
    if camera_id not in VIDEO_POOLS:
        return jsonify({"error": "Camera not found"}), 404

    return Response(
        generate_frames(camera_id),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/status')
def status():
    return jsonify(build_status_summary())

if __name__ == '__main__':
    VIDEO_POOLS = {}
    for camera_id, source_videos in SOURCE_POOLS.items():
        playlist = source_videos[:]
        random.shuffle(playlist)
        VIDEO_POOLS[camera_id] = playlist

    total_sources = sum(len(videos) for videos in VIDEO_POOLS.values())
    print(f"Source clips ready: {total_sources}")
    try:
        try:
            from waitress import serve
            print("Starting server with waitress...")
            serve(app, host='0.0.0.0', port=5000, threads=8)
        except Exception as server_error:
            print(f"Waitress unavailable ({server_error}), falling back to Flask dev server")
            app.run(host='0.0.0.0', port=5000, threaded=True)
    except KeyboardInterrupt:
        print("\nShutdown requested. Cleaning up...")
    finally:
        cleanup_runtime()
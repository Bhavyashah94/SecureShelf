import argparse
import os
from glob import glob

import cv2
import joblib
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO


DATASET_DIR = r"C:\Users\bhavy\Documents\Project\SecureShelf\Dataset"
DATA_DIR = r"C:\Users\bhavy\Documents\Project\SecureShelf\ProcessedData"
MODEL_PATH = os.path.join(DATA_DIR, "best_v4.pth")
SCALER_PATH = os.path.join(DATA_DIR, "scaler_v4.pkl")
POSE_MODEL_PATH = r"C:\Users\bhavy\Documents\Project\SecureShelf\yolo26n-pose.pt"

INPUT_FEATURES = 378
MAX_FRAMES = 40
MAX_PEOPLE = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cos_theta = np.dot(v1, v2) / (n1 * n2)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))


def get_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def extract_frame_features(frame, yolo_results):
    frame_data = np.zeros((MAX_PEOPLE, 63))

    if not yolo_results or yolo_results[0].boxes is None or len(yolo_results[0].boxes) == 0:
        return frame_data.flatten()

    results = yolo_results[0]
    keypoints = results.keypoints.data.cpu().numpy() if hasattr(results, "keypoints") else None
    boxes = results.boxes.xyxy.cpu().numpy() if hasattr(results.boxes, "xyxy") else None

    if keypoints is None or boxes is None:
        return frame_data.flatten()

    for slot in range(min(len(keypoints), MAX_PEOPLE)):
        kp = keypoints[slot]
        box = boxes[slot]

        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        diag = np.sqrt((box[2] - box[0]) ** 2 + (box[3] - box[1]) ** 2) + 1e-6

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


def classify_pose_sequence(sequence, model, scaler):
    if len(sequence) < MAX_FRAMES:
        return 0

    x_batch = np.array(sequence[-MAX_FRAMES:])
    x_reshaped = x_batch.reshape(MAX_FRAMES, MAX_PEOPLE, 63)

    raw_features = x_reshaped
    velocity = np.zeros_like(raw_features)
    velocity[1:] = raw_features[1:] - raw_features[:-1]
    final_sequence = np.concatenate([raw_features, velocity], axis=2)

    final_sequence_flat = final_sequence.reshape(MAX_FRAMES, MAX_PEOPLE * 126)
    x_scaled = scaler.transform(final_sequence_flat)
    x_tensor = torch.FloatTensor(x_scaled).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1)
        return float(probs[0, 1].item()) * 100.0


def run_video(video_path, yolo_model, model, scaler, threshold, detect_stride, show_window):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open: {video_path}")
        return None

    pose_sequence = []
    frame_count = 0
    max_prob = 0.0
    detection_events = 0
    classifications = 0
    last_prob = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        do_detect = frame_count % max(1, detect_stride) == 0
        results = yolo_model.predict(frame, verbose=False) if do_detect else []

        frame_features = extract_frame_features(frame, results)
        pose_sequence.append(frame_features)
        if len(pose_sequence) > MAX_FRAMES * 2:
            pose_sequence = pose_sequence[-MAX_FRAMES:]

        theft_prob = last_prob
        if frame_count % 10 == 0 and len(pose_sequence) >= MAX_FRAMES:
            theft_prob = classify_pose_sequence(pose_sequence, model, scaler)
            last_prob = theft_prob
            classifications += 1
            max_prob = max(max_prob, theft_prob)
            if theft_prob >= threshold:
                detection_events += 1

        if show_window:
            label = "DETECTED" if theft_prob >= threshold else "NOT DETECTED"
            color = (0, 0, 255) if theft_prob >= threshold else (0, 200, 0)
            vis = frame.copy()
            cv2.putText(
                vis,
                f"Theft Prob: {theft_prob:5.1f}% | {label}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                f"Frame: {frame_count}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Detection Debug", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if show_window:
        cv2.destroyAllWindows()

    return {
        "video": video_path,
        "frames": frame_count,
        "classifications": classifications,
        "max_prob": max_prob,
        "last_prob": last_prob,
        "detected": detection_events > 0,
        "events": detection_events,
    }


def collect_videos(target_video, target_class, limit):
    if target_video:
        return [target_video]

    classes = ["Normal", "Shoplifting"] if target_class == "all" else [target_class]
    videos = []
    for cls in classes:
        pattern = os.path.join(DATASET_DIR, cls, "*.mp4")
        videos.extend(sorted(glob(pattern)))

    if limit > 0:
        videos = videos[:limit]

    return videos


def main():
    parser = argparse.ArgumentParser(description="Debug theft detection outside web UI")
    parser.add_argument("--video", type=str, default="", help="Path to a single video")
    parser.add_argument(
        "--class",
        dest="target_class",
        choices=["Normal", "Shoplifting", "all"],
        default="Shoplifting",
        help="Dataset class to test when --video is not provided",
    )
    parser.add_argument("--limit", type=int, default=5, help="How many videos to test")
    parser.add_argument("--threshold", type=float, default=50.0, help="Alert threshold percent")
    parser.add_argument("--stride", type=int, default=2, help="Pose detection stride")
    parser.add_argument("--show", action="store_true", help="Show live window for video")
    args = parser.parse_args()

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Missing model: {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Missing scaler: {SCALER_PATH}")

    print("Loading models...")
    yolo_model = YOLO(POSE_MODEL_PATH if os.path.exists(POSE_MODEL_PATH) else "yolo26n-pose.pt")
    model = PoseGRU().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    scaler = joblib.load(SCALER_PATH)

    videos = collect_videos(args.video, args.target_class, args.limit)
    if not videos:
        print("No videos found to test.")
        return

    print(f"Testing {len(videos)} video(s) | threshold={args.threshold:.1f}%")
    print("-" * 80)

    detected_count = 0
    for idx, video in enumerate(videos, start=1):
        result = run_video(
            video_path=video,
            yolo_model=yolo_model,
            model=model,
            scaler=scaler,
            threshold=args.threshold,
            detect_stride=args.stride,
            show_window=args.show,
        )

        if result is None:
            continue

        detected = "YES" if result["detected"] else "NO"
        if result["detected"]:
            detected_count += 1

        print(
            f"[{idx:02d}] {os.path.basename(result['video'])} | "
            f"max_prob={result['max_prob']:.1f}% | last_prob={result['last_prob']:.1f}% | "
            f"events={result['events']} | detected={detected}"
        )

    print("-" * 80)
    print(f"Detected in {detected_count}/{len(videos)} videos")
    print("Tip: Lower --threshold to 40 for sensitivity testing.")


if __name__ == "__main__":
    main()
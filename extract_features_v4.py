import os
import cv2
import numpy as np
from ultralytics import YOLO

# --- Configuration ---
DATA_DIR = r"C:\Users\bhavy\Documents\Project\SecureShelf\Dataset"
SAVE_DIR = r"C:\Users\bhavy\Documents\Project\SecureShelf\ProcessedData"
CLASSES = ['Normal', 'Shoplifting']
MAX_FRAMES = 40
MAX_PEOPLE = 3

os.makedirs(SAVE_DIR, exist_ok=True)
print("Loading YOLO Model...")
model = YOLO('yolo26n-pose.pt')

def get_angle(p1, p2, p3):
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]])
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0: return 0.0
    cos_theta = np.dot(v1, v2) / (n1 * n2)
    return np.arccos(np.clip(cos_theta, -1.0, 1.0))

def get_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def extract_v4(video_path):
    cap = cv2.VideoCapture(video_path)
    skip = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // MAX_FRAMES)
    
    features = []
    track_history = {} 
    next_slot = 0
    frame_count = 0
    
    while cap.isOpened() and len(features) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret: break
            
        if frame_count % skip == 0:
            results = model.track(frame, persist=True, verbose=False)
            
            # Each person: 17x2 (normalized x,y) + 17 (confidence) = 51
            # + 8 angles + 4 distances = 63 features per person * 3 = 189
            frame_data = np.zeros((MAX_PEOPLE, 63))
            
            if len(results[0].boxes) > 0 and results[0].boxes.id is not None:
                ids = results[0].boxes.id.int().cpu().tolist()
                keypoints = results[0].keypoints.data.cpu().numpy()
                boxes = results[0].boxes.xyxy.cpu().numpy()
                
                for i, track_id in enumerate(ids):
                    if track_id not in track_history:
                        if next_slot < MAX_PEOPLE:
                            track_history[track_id] = next_slot
                            next_slot += 1
                        else: continue 
                            
                    slot = track_history[track_id]
                    if len(keypoints) > i: 
                        kp = keypoints[i] # 17x3
                        box = boxes[i] # x1, y1, x2, y2
                        
                        # Calculate center and diagonal for normalization
                        cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
                        diag = np.sqrt((box[2] - box[0])**2 + (box[3] - box[1])**2) + 1e-6
                        
                        norm_kp = np.zeros((17, 3))
                        for j in range(17):
                            if kp[j][2] > 0.1: # if confident
                                norm_kp[j][0] = (kp[j][0] - cx) / diag
                                norm_kp[j][1] = (kp[j][1] - cy) / diag
                                norm_kp[j][2] = kp[j][2]
                                
                        frame_data[slot][:51] = norm_kp.flatten()
                        
                        # Angles
                        if len(kp) >= 17:
                            angles = [
                                get_angle(kp[5], kp[7], kp[9]),   # L Arm
                                get_angle(kp[6], kp[8], kp[10]),  # R Arm
                                get_angle(kp[11], kp[13], kp[15]),# L Leg
                                get_angle(kp[12], kp[14], kp[16]),# R Leg
                                get_angle(kp[7], kp[5], kp[11]),  # L Shoulder-Hip
                                get_angle(kp[8], kp[6], kp[12]),  # R Shoulder-Hip
                                get_angle(kp[5], kp[11], kp[13]), # L Hip-Knee
                                get_angle(kp[6], kp[12], kp[14])  # R Hip-Knee
                            ]
                            
                            distances = [
                                get_dist(norm_kp[9][:2], norm_kp[11][:2]), # L Wrist to L Hip
                                get_dist(norm_kp[10][:2], norm_kp[12][:2]),# R Wrist to R Hip
                                get_dist(norm_kp[9][:2], norm_kp[10][:2]), # L Wrist to R Wrist
                                get_dist(norm_kp[5][:2], norm_kp[9][:2])   # L Shoulder to L Wrist
                            ]
                            
                            frame_data[slot][51:59] = angles
                            frame_data[slot][59:63] = distances
                            
            features.append(frame_data.flatten())
            
        frame_count += 1
    cap.release()
    
    # Pad if too short
    while len(features) < MAX_FRAMES:
        features.append(np.zeros(MAX_PEOPLE * 63))
        
    raw_features = np.array(features)
    
    # Calculate Velocity (difference between current frame and previous frame)
    velocity = np.zeros_like(raw_features)
    velocity[1:] = raw_features[1:] - raw_features[:-1]
    
    # Concatenate: Base (189) + Velocity (189) = 378
    final_sequence = np.concatenate([raw_features, velocity], axis=1)
    return final_sequence

if __name__ == "__main__":
    X, y = [], []
    for label, cls in enumerate(CLASSES):
        cls_path = os.path.join(DATA_DIR, cls)
        for video_name in os.listdir(cls_path):
            if video_name.endswith(('.mp4', '.avi')):
                print(f"Extraction V4: {cls} -> {video_name}")
                seq = extract_v4(os.path.join(cls_path, video_name))
                X.append(seq)
                y.append(label)

    X, y = np.array(X), np.array(y)
    print(f"\nExtraction V4 Complete! Shape: {X.shape}") 
    np.save(os.path.join(SAVE_DIR, 'X_features_v4.npy'), X)
    np.save(os.path.join(SAVE_DIR, 'y_labels_v4.npy'), y)

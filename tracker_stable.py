import cv2
import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from ultralytics import YOLO
import numpy as np
import json
from scipy.spatial.distance import cosine
from collections import deque


VIDEO_FILES = {
    "tacticam": "src/tacticam.mp4",
    "broadcast": "src/broadcast.mp4"
}
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.3
SIM_THRESHOLD = 0.4
TRACK_HISTORY = 5

os.makedirs("tracker_logs_stable", exist_ok=True)
os.makedirs("output_with_ids", exist_ok=True)

# --- Load models ---
device = "cuda" if torch.cuda.is_available() else "cpu"
detector = YOLO("weights/best.pt").to(device)
resnet = resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval().to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
    return interArea / float(box1Area + box2Area - interArea)


for video_tag, video_path in VIDEO_FILES.items():
    output_json = f"tracker_logs_stable/{video_tag}.json"
    output_video_path = f"output_with_ids/{video_tag}_tracked.mp4"
    next_id = 1
    active_tracks = []  
    output = []

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    if not cap.isOpened():
        print(f" Could not open video: {video_path}")
        continue

    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print(f" Starting stable tracking for: {video_tag}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = detector(frame)[0]

        for box in results.boxes:
            conf = box.conf.item()
            if conf < CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            with torch.no_grad():
                tensor = transform(crop).unsqueeze(0).to(device)
                feature = resnet(tensor).squeeze().cpu().numpy()

            matched = False
            best_sim = 1.0
            best_track = None

            for track in active_tracks:
                iou_val = iou([x1, y1, x2, y2], track["bbox"])
                avg_feat = np.mean(track["features"], axis=0)
                sim = cosine(feature, avg_feat)
                if iou_val > IOU_THRESHOLD and sim < SIM_THRESHOLD and sim < best_sim:
                    matched = True
                    best_sim = sim
                    best_track = track

            if matched:
                best_track["bbox"] = [x1, y1, x2, y2]
                best_track["features"].append(feature)
                if len(best_track["features"]) > TRACK_HISTORY:
                    best_track["features"].popleft()
                track_id = best_track["id"]
            else:
                track_id = next_id
                next_id += 1
                active_tracks.append({
                    "id": track_id,
                    "bbox": [x1, y1, x2, y2],
                    "features": deque([feature], maxlen=TRACK_HISTORY)
                })

            output.append({
                "frame": frame_idx,
                "id": track_id,
                "bbox": [x1, y1, x2, y2]
            })

           
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    with open(output_json, "w") as f:
        json.dump(output, f, indent=2)

    print(f" Finished tracking {video_tag} — {frame_idx} frames, {len(output)} detections saved")
    print(f" Output JSON saved: {output_json}")
    print(f" Video saved: {output_video_path}")

import cv2
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import json
from collections import defaultdict


videos = {
    "broadcast": r"output_with_ids\broadcast_tracked.mp4",
    "tacticam": r"output_with_ids\tacticam_tracked.mp4"
}
output_dir = "updated_features"
os.makedirs(output_dir, exist_ok=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval().to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def extract_id_from_region(region):
    region = cv2.threshold(region, 200, 255, cv2.THRESH_BINARY)[1]
    digits = [chr(c) for c in region.flatten() if 48 <= c <= 57]
    digit_str = ''.join(digits)
    return int(digit_str) if digit_str.isdigit() else None


for tag, video_path in videos.items():
    cap = cv2.VideoCapture(video_path)
    features_per_id = defaultdict(list)
    frame_idx = 0
    print(f"[⏳] Processing {tag}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     
        lower_green = np.array([36, 25, 25])
        upper_green = np.array([86, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            roi = frame[y-25:y, x:x+80] 
            if roi.shape[0] <= 0 or roi.shape[1] <= 0:
                continue

            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            id_val = extract_id_from_region(gray_roi)
            if id_val is None:
                continue

            crop = frame[y:y+h, x:x+w]
            if crop.size == 0:
                continue

            with torch.no_grad():
                tensor = transform(crop).unsqueeze(0).to(device)
                feature = resnet(tensor).squeeze().cpu().numpy()
                features_per_id[id_val].append(feature)

        frame_idx += 1

    cap.release()

   
    avg_features = []
    id_list = []
    for id_val, feats in features_per_id.items():
        if len(feats) >= 3:
            avg_feat = np.mean(np.array(feats), axis=0)
            avg_features.append(avg_feat)
            id_list.append(id_val)

    np.save(os.path.join(output_dir, f"{tag}.npy"), np.array(avg_features))
    with open(os.path.join(output_dir, f"{tag}_ids.json"), "w") as f:
        json.dump(id_list, f)

    print(f" Done: {tag}.npy and {tag}_ids.json saved.")

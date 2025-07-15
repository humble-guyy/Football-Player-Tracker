import cv2
import json
import os

os.makedirs("final_output", exist_ok=True)

tracker_logs = {
    "broadcast": "tracker_logs_stable/broadcast.json",
    "tacticam": "tracker_logs_stable/tacticam.json"
}
input_videos = {
    "broadcast": "src/broadcast.mp4",
    "tacticam": "src/tacticam.mp4"
}

for tag in ["broadcast", "tacticam"]:
    print(f"[📄] Loading log for: {tag}")
    with open(tracker_logs[tag], "r") as f:
        tracked = json.load(f)

    cap = cv2.VideoCapture(input_videos[tag])
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out_path = f"final_output/{tag}_final.avi"

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    if not writer.isOpened():
        print(f" ERROR: Could not open VideoWriter for {out_path}")
        continue

    print(f" Writing final video: {tag}")

    frame_data = {}
    for entry in tracked:
        frame_idx = entry["frame"]
        if frame_idx not in frame_data:
            frame_data[frame_idx] = []
        frame_data[frame_idx].append(entry)

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for obj in frame_data.get(idx, []):
            x1, y1, x2, y2 = obj["bbox"]
            gid = obj["id"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"GID {gid}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
    print(f" Done: {out_path}")

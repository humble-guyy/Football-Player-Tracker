# Football-Player-Tracker


# Cross-Camera Player Mapping (Football AI)

It is an end-to-end computer vision system to detect, track, and identify football players across multiple camera angles.  
It uses YOLOv11 for object detection, ResNet50 for appearance feature extraction, and custom logic to assign consistent global IDs across unsynced video feeds.

---

##  Use Case

Given two input videos of the same football match — one from a **broadcast camera** and another from a **tacticam** — the system:

- Detects and tracks players within each video
- Assigns stable **local IDs** using appearance and position
- Extracts **ResNet-based features** for each player
- Matches the same players across both videos
- Generates final output videos with consistent **global IDs**

---

##  Folder Structure

Football-Player-Tracker/

├── src/  # Raw input videos (broadcast.mp4, tacticam.mp4)

├── weights/ # Your trained YOLOv11 model (best.pt)

├── output_with_ids/tracker/ # Videos with local ID tracking drawn

├── tracker_logs_stable/ # JSON tracker logs with stable IDs

├── updated_features/ # ResNet features, ID mappings, global_ids.json

├── final_output/ # Final videos with consistent Global IDs

├── main.py # One-click runner script for full pipeline

├── requirements.txt # Python dependencies

└── README.md



---

##  Setup Instructions

### 1.  Clone the Repository

```bash
git clone https://github.com/humble-guyy/Football-Player-Tracker.git
cd Football-Player-Tracker
```

### 2. Create a Virtual Environment


```bash
python -m venv venv
venv\\Scripts\\activate        # On Windows
# OR
source venv/bin/activate       # On macOS/Linux
```
### 3. Install Dependencies
```bash

pip install -r requirements.txt
```

### 4. Download the model
```bash

 python download_weights.py
```
### 5. Run the Full Pipeline
```bash
python main.py
```

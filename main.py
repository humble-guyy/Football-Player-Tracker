import subprocess

# 1️⃣ Track players in raw videos → saves .json logs + tracked videos
subprocess.run(["python", "tracker_stable.py"])

# 2️⃣ Extract ResNet features for each stable-tracked player
subprocess.run(["python", "extract_id_features.py"])

# 3️⃣ Match players across cameras using those features
subprocess.run(["python", "cross_camera_matcher.py"])

# 4️⃣ Debug how many global IDs matched correctly
subprocess.run(["python", "global_id_debugger.py"])

# 5️⃣ Draw final videos using consistent global IDs
subprocess.run(["python", "rewrite_video_with_global_ids.py"])

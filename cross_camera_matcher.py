import numpy as np
import json
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment
import os

# --- Paths ---
feature_dir = "updated_features"
output_path = os.path.join(feature_dir, "global_ids.json")

# --- Load embeddings and IDs ---
broadcast_feats = np.load(os.path.join(feature_dir, "broadcast.npy"))
tacticam_feats = np.load(os.path.join(feature_dir, "tacticam.npy"))

with open(os.path.join(feature_dir, "broadcast_ids.json")) as f:
    broadcast_ids = json.load(f)

with open(os.path.join(feature_dir, "tacticam_ids.json")) as f:
    tacticam_ids = json.load(f)

# --- Build similarity matrix (lower = more similar) ---
cost_matrix = np.zeros((len(broadcast_feats), len(tacticam_feats)))
for i, b in enumerate(broadcast_feats):
    for j, t in enumerate(tacticam_feats):
        cost_matrix[i][j] = cosine(b, t)

# --- Optimal assignment ---
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# --- Build global ID map ---
global_id = 1
global_id_map = {}

for b_idx, t_idx in zip(row_ind, col_ind):
    sim_score = 1 - cost_matrix[b_idx][t_idx]
    if sim_score < 0.7:  # Only accept strong matches
        continue
    b_id = broadcast_ids[b_idx]
    t_id = tacticam_ids[t_idx]
    global_id_map[("broadcast", b_id)] = global_id
    global_id_map[("tacticam", t_id)] = global_id
    global_id += 1

# --- Save map ---
with open(output_path, "w") as f:
    json.dump(global_id_map, f, indent=2)

print(f"[✓] Global ID map saved to {output_path}")

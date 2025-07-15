
import json
from collections import defaultdict

# Paths
global_id_path = "updated_features/global_ids.json"
stable_logs = {
    "broadcast": "tracker_logs_stable/broadcast.json",
    "tacticam": "tracker_logs_stable/tacticam.json"
}


with open(global_id_path) as f:
    global_ids = json.load(f)

global_keys = set(global_ids.keys())
global_reverse = defaultdict(list)

for k, v in global_ids.items():
    global_reverse[v].append(k)


for tag, log_path in stable_logs.items():
    with open(log_path) as f:
        logs = json.load(f)

    all_ids = set(entry["id"] for entry in logs)
    matched_ids = set()
    for pid in all_ids:
        key = json.dumps([tag, pid])
        if key in global_ids:
            matched_ids.add(pid)

    print(f"--- {tag.upper()} ---")
    print(f"Total local IDs:     {len(all_ids)}")
    print(f"Matched to global:   {len(matched_ids)}")
    print(f"Unmatched IDs:       {sorted(all_ids - matched_ids)}")
    print()

print(" GLOBAL ID OVERLAPS:")
for gid, keys in global_reverse.items():
    if len(keys) > 2:
        print(f"[] Global ID {gid} has {len(keys)} mappings → {keys}")

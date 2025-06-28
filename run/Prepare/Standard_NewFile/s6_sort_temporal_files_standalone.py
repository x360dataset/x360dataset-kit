import os
import json
from class_label import class_mapping, temporal_mapping
import re

# Set your base path here
base_path = '/Users/haochen/Downloads/360x_dataset_demo/TAL_annotations'

# Read hash mapping
import os
import json
from class_label import class_mapping, temporal_mapping
import re


# Read hash mapping

# Prepare output lists
temporal_lines = [("id", "label", "Usage")]

# Walk through directories
for folder in os.listdir(base_path):

    print(f"Processing folder: {folder}")
    folder = folder.replace(".json", "")  # Ensure folder name does not end with .json

    # ---------- Temporal Label ----------
    temporal_json_path = os.path.join(base_path, f"{folder}.json")
    print(f"Processing temporal json: {temporal_json_path}")

    if os.path.exists(temporal_json_path):


        with open(temporal_json_path, 'r', encoding='utf-8', errors='replace') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️  Warning: Failed to parse JSON for {temporal_json_path}. Skipping.")
                continue

        metadata = data.get("metadata", {})
        action_entries = []

        for key, meta in metadata.items():
            duration = meta.get("duration")
            action_dict = meta.get("action", {})

            if not duration or "1" not in action_dict:
                continue

            action_name = action_dict["1"]
            if action_name not in temporal_mapping:
                print(f"⚠️  Warning: Action '{action_name}' not in temporal_mapping. Skipping entry.")
                continue

            class_number = temporal_mapping[action_name]
            start_time, end_time = duration
            action_entries.append([class_number, start_time, end_time])

        if action_entries:
            hashed_id = folder
            temporal_label_str = json.dumps(action_entries)
            temporal_lines.append((hashed_id, f'"{temporal_label_str}"', "Private"))
            print(f"✔️  Added temporal label for {hashed_id}")

print("\nWriting temporal_final_label_file.csv...")
temporal_csv_path = os.path.join(base_path, "..", "temporal_final_label_file.csv")
with open(temporal_csv_path, 'w') as f:
    for line in temporal_lines:
        f.write(f"{line[0]},{line[1]},{line[2]}\n")
print(f"✔️  Temporal file written to {temporal_csv_path}")




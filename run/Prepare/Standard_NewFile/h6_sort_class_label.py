import os
import json
from class_label import class_mapping, temporal_mapping
import re

# Set your base path here
base_path = '/bask/projects/j/jiaoj-rep-learn/360XProject/ICCV_Workshop_Data/Han/label_unanonymized_published_data'

# Read hash mapping
hash_mapping_path = os.path.join(base_path, "..", "hash_mapping.txt")
hash_dict = {}

with open(hash_mapping_path, 'r') as f:
    for line in f:
        if ':' in line:
            hashed, orig = line.strip().split(':', 1)
            hash_dict[orig.strip()] = hashed.strip()

print(f"Loaded {len(hash_dict)} hash mappings.")

# Prepare output lists
output_lines = [("id", "label", "Usage")]
temporal_lines = [("id", "label", "Usage")]

# Walk through directories
for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)

    if not os.path.isdir(folder_path):
        continue

    print(f"Processing folder: {folder}")

    # ---------- Classification Label ----------
    label_json_path = os.path.join(folder_path, "label.json")
    if os.path.exists(label_json_path):
        with open(label_json_path, 'r', encoding='utf-8', errors='replace') as f:
            label_name = f.read().strip()

        if label_name not in class_mapping:
            print(f"⚠️  Warning: Label '{label_name}' not in class_mapping. Skipping classification label.")
        elif folder not in hash_dict:
            print(f"⚠️  Warning: Folder '{folder}' not found in hash_mapping. Skipping classification label.")
        else:
            class_number = class_mapping[label_name]
            hashed_id = hash_dict[folder]
            output_lines.append((hashed_id, class_number, "Private"))
            print(f"✔️  Added classification label for {hashed_id}")

    # ---------- Temporal Label ----------
    temporal_json_path = os.path.join(folder_path, f"{folder}.json")
    if os.path.exists(temporal_json_path):
        if folder not in hash_dict:
            print(f"⚠️  Warning: Folder '{folder}' not in hash_mapping. Skipping temporal label.")
            continue

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
            hashed_id = hash_dict[folder]
            temporal_label_str = json.dumps(action_entries)
            temporal_lines.append((hashed_id, f'"{temporal_label_str}"', "Private"))
            print(f"✔️  Added temporal label for {hashed_id}")

print("\nWriting classification_final_label_file.csv...")
output_csv_path = os.path.join(base_path, "..", "classification_final_label_file.csv")
with open(output_csv_path, 'w') as f:
    for line in output_lines:
        f.write(f"{line[0]},{line[1]},{line[2]}\n")
print(f"✔️  Classification file written to {output_csv_path}")

print("\nWriting temporal_final_label_file.csv...")
temporal_csv_path = os.path.join(base_path, "..", "temporal_final_label_file.csv")
with open(temporal_csv_path, 'w') as f:
    for line in temporal_lines:
        f.write(f"{line[0]},{line[1]},{line[2]}\n")
print(f"✔️  Temporal file written to {temporal_csv_path}")

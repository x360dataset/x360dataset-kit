import pandas as pd
import ast
import math
from collections import defaultdict

class_csv = "/Volumes/Elements/360_ICCV_Workshop/temporal_final_label_file.csv"

# Read the file
df = pd.read_csv(class_csv)


public_indices = df.sample(frac=0.5, random_state=42).index
df['Usage'] = "Private"
df.loc[public_indices, 'Usage'] = "Public"


# Use a dictionary to group labels by (id + time range) -> store list of labels and usage
grouped_labels = defaultdict(lambda: {"labels": [], "usage": None})


for idx, row in df.iterrows():
    base_id = row['id']
    usage = row.get('Usage', 'Private')  # Default to 'Private' if missing

    try:
        label_list = ast.literal_eval(row['label'])
    except:
        continue

    if not isinstance(label_list, list):
        continue

    for label in label_list:
        if not isinstance(label, list) or len(label) < 3:
            continue

        cls, start, end = label[:3]

        start_bin = math.floor(start / 10) * 10
        end_bin = math.ceil(end / 10) * 10

        for bin_start in range(start_bin, end_bin, 10):
            bin_end = bin_start + 10

            seg_start = max(start, bin_start)
            seg_end = min(end, bin_end)

            length = seg_end - seg_start

            if seg_start < seg_end and length >= 0.5:
                key = f"{base_id}_{int(bin_start)}_{int(bin_end)}"
                grouped_labels[key]["labels"].append([cls, round(seg_start, 1), round(seg_end, 1)])
                grouped_labels[key]["usage"] = usage  # Store usage per group

# Convert grouped results to DataFrame
output_rows = []

for key, group in grouped_labels.items():
    output_rows.append({
        "id": key,
        "label": str(group["labels"]),
        "Usage": group["usage"]
    })

output_df = pd.DataFrame(output_rows)

print(output_df.head())

# Optional: Save to CSV
output_df.to_csv("/Volumes/Elements/360_ICCV_Workshop/temporal_final_label_file_split_10s.csv", index=False)
print("✔️  temporal_final_label_file_split_10s.csv saved.")

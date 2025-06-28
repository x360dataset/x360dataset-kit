import pandas as pd
import ast
import random
import numpy as np


class_csv = "/Volumes/Elements/360_ICCV_Workshop/temporal_final_label_file_split_10s.csv"

# Read with header
df = pd.read_csv(class_csv)



print(df.head())  # Sanity check

# Remove the 'Usage' column

df = df.drop(columns=["Usage"])

# Rename the 'label' column to 'prediction'
df = df.rename(columns={"label": "prediction"})

# Process the 'prediction' column
def process_prediction(pred_str):

    try:
        preds = ast.literal_eval(pred_str)  # Safely parse the string to a list
        if not isinstance(preds, list):
            return "[]"


        # Append score 0.999 to each
        processed = [p + [0.999] for p in preds]

        return str(processed)
    except:
        return "[]"

df['prediction'] = df['prediction'].apply(process_prediction)


print(df.head())  # Check result

# Save to CSV
df.to_csv("/Volumes/Elements/360_ICCV_Workshop/temporal_full_sample_submission.csv", index=False)
print("✔️  temporal_sample_submission.csv saved.")





#  --------

# Read with header
df = pd.read_csv(class_csv)

print(df.head())  # Sanity check

# Remove 'Usage' and rename 'label' to 'prediction'
df = df.drop(columns=["Usage"])
df = df.rename(columns={"label": "prediction"})

# Process the 'prediction' column
def process_prediction(row):
    pred_str = row['prediction']
    id_str = row['id']

    # Extract time range from ID
    try:
        parts = id_str.split('_')
        start_time = float(parts[-2])
        end_time = float(parts[-1])
    except:
        start_time = 0
        end_time = 9999  # fallback if parsing fails

    try:
        preds = ast.literal_eval(pred_str)
        if not isinstance(preds, list):
            return "[]"

        # Randomly sample up to 2 entries
        sampled = random.sample(preds, min(2, len(preds)))
        if np.random.random() < 0.5:
            sampled = []

        processed = []
        for p in sampled:
            cls, start, end = p[:3]

            start_shift = round(random.uniform(-1.5, 1.5), 1)
            end_shift = round(random.uniform(-1.5, 1.5), 1)

            new_start = round(start + start_shift, 1)
            new_end = round(end + end_shift, 1)

            new_start = max(start_time, new_start)
            new_end = min(end_time, max(new_start + 0.1, new_end))

            if new_start > new_end:
                new_start, new_end = new_end, new_start

            processed.append([cls, new_start, new_end, 0.999])

        num_classes = 35

        # Add 1-3 random guest predictions within current time bin
        num_guests = np.random.randint(1, 2)
        for _ in range(num_guests):
            cls = random.randint(0, num_classes - 1)
            guest_start = round(random.uniform(start_time, end_time - 0.5), 1)
            guest_end = round(guest_start + random.uniform(0.5, min(20, end_time - guest_start)), 1)
            guest_end = min(guest_end, end_time)

            if guest_start < guest_end:
                processed.append([cls, guest_start, guest_end, 0.999])

        return str(processed)
    except:
        return "[]"

df['prediction'] = df.apply(process_prediction, axis=1)

print(df.head())  # Check result

# Save to CSV
df.to_csv("/Volumes/Elements/360_ICCV_Workshop/temporal_sample_submission.csv", index=False)
print("✔️  temporal_sample_submission.csv saved.")

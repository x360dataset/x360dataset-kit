import pandas as pd
import random
class_csv = "/Volumes/Elements/360_ICCV_Workshop/classification_final_label_file.csv"

# Read with header
df = pd.read_csv(class_csv)

print(df.head())  # Sanity check


public_indices = df.sample(frac=0.5, random_state=42).index  # random_state for reproducibility

# Set all rows to "Private" by default
df['Usage'] = "Private"
# Set sampled rows to "Public"
df.loc[public_indices, 'Usage'] = "Public"
df.to_csv("/Volumes/Elements/360_ICCV_Workshop/classification_final_label_file_with_usage.csv", index=False)




# Get unique class labels
num_classes = 28 # df['label'].max() + 1  # Assuming class labels are integers from 0 upwards

# Initialize one-hot columns
for i in range(num_classes):
    df[f'{i}'] = 0

# One-hot encode
for idx, row in df.iterrows():
    class_idx = int(row['label'])
    if 0 <= class_idx < num_classes:
        df.at[idx, f'{class_idx}'] = 1
    else:
        print(f"⚠️  Warning: Unexpected class index '{class_idx}' in row {idx}. Skipping.")

# Keep only id and one-hot columns
one_hot_df = df[['id'] + [f'{i}' for i in range(num_classes)]]

# Save to new CSV
output_path = "/Volumes/Elements/360_ICCV_Workshop/classification_full_sample_submission.csv"
one_hot_df.to_csv(output_path, index=False)

print(f"✔️  One-hot encoded file written to {output_path}")



# one_hot_df, leave only 25% correct in the sample submission
# ---------- Generate Sample Submission with 25% Correct ----------
sample_df = one_hot_df.copy()

# Randomly select 75% of rows to corrupt
corrupt_indices = sample_df.sample(frac=0.75, random_state=123).index

for idx in corrupt_indices:
    correct_class = sample_df.loc[idx, sample_df.columns[1:]].idxmax()
    correct_col = correct_class

    # Set all class columns to 0
    for col in sample_df.columns[1:]:
        sample_df.at[idx, col] = 0

    # Randomly select a wrong class
    possible_classes = [col for col in sample_df.columns[1:] if col != correct_col]
    wrong_class = random.choice(possible_classes)
    sample_df.at[idx, wrong_class] = 1

sample_submission_path = "/Volumes/Elements/360_ICCV_Workshop/classification_sample_submission.csv"
sample_df.to_csv(sample_submission_path, index=False)
print(f"✔️  Sample submission with 25% correct saved to {sample_submission_path}")

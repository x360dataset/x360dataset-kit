import os
import hashlib
import pickle

# Path to your main directory (adjust accordingly)
base_path = '/bask/projects/j/jiaoj-rep-learn/360XProject/ICCV_Workshop_Data/Han/unanonymized_published_data'  # Change this to your actual path

# Dictionary to store hash -> original name mapping
hash_mapping = {}


# Function to create a 16-character hash from a folder name
def hash_name(name):
    hash_obj = hashlib.sha256(name.encode('utf-8'))
    hex_digest = hash_obj.hexdigest()

    # Format as: 8-4-4-4-12 characters (like UUID)
    formatted_hash = f"{hex_digest[:8]}-{hex_digest[8:12]}-{hex_digest[12:16]}-{hex_digest[16:20]}-{hex_digest[20:32]}"

    return formatted_hash


hash_mapping_path = os.path.join(base_path, "hash_mapping.pkl")
hash_mapping = {}

if os.path.exists(hash_mapping_path):
    with open(hash_mapping_path, 'rb') as f:
        hash_mapping = pickle.load(f)
    print("Loaded existing hash_mapping.")
    have_hash = True
else:
    print("No existing hash_mapping found. Creating new mapping.")
    have_hash = False

reverse_mapping = {v: k for k, v in hash_mapping.items()} if have_hash else {}


# Iterate over all top-level folders
for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)

    if os.path.isdir(folder_path):
        hashed = hash_name(folder)

        new_folder_path = os.path.join(base_path, hashed)

        print(f"Renaming '{folder}' -> '{hashed}'")

        # effectively renaming the folder
        os.rename(folder_path, new_folder_path)

        if not have_hash:
            hash_mapping[hashed] =   folder
            hash_mapping[folder] =   hashed
        else:
            if folder in reverse_mapping:
                old_hashed = reverse_mapping[folder]
                hash_mapping[hashed] = hash_mapping[old_hashed]
                hash_mapping[folder] = old_hashed
            else:
                hash_mapping[hashed] = folder
                hash_mapping[folder] = old_hashed



# Save mapping to pickle
pickle_path = os.path.join(base_path + "/..", 'hash_mapping.pkl')

with open(pickle_path, 'wb') as f:
    pickle.dump(hash_mapping, f)

txt_path = os.path.join(base_path + "/..", 'hash_mapping.txt')
with open(txt_path, 'w') as f:
    for hashed, original in hash_mapping.items():
        f.write(f"{hashed} : {original}\n")

print(f"\nMapping saved to: {pickle_path}")

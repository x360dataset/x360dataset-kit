import os

folder = "/Volumes/Elements/360_ICCV_Workshop/360x_Classification_Val"  # Replace with your target folder

# Walk through all directories and subdirectories
for root, dirs, files in os.walk(folder):
    for filename in files:
        if filename.startswith("._"):
            file_path = os.path.join(root, filename)
            try:
                os.remove(file_path)
                print(f"Removed: {file_path}")
            except Exception as e:
                print(f"Failed to remove {file_path}: {e}")



    for filename in files:
        if ".DS_Store" in filename:
            file_path = os.path.join(root, filename)
            try:
                os.remove(file_path)
                print(f"Removed: {file_path}")
            except Exception as e:
                print(f"Failed to remove {file_path}: {e}")
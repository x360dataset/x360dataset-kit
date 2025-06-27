
# ----------- This steps if to sort the files in the new standard format for cloud saving -----------
python s0_analysis.py      --root_path  /Volumes/Elements/2024_ready/

# Rename the 360 to 360_panoramic and front_view
python s1_rename_360.py      --root_path  /Volumes/Elements/2024_ready/

# Move snapchat files to the new folder clip1, clip2, etc.
python s2_format_snapchat.py --root_path /Volumes/Elements/2024_ready_copy/



python s3_process_snapchat.py --root_path /Volumes/Elements/2024_ready/



# But this use database
python extract_data.py  + the following action you want
			 --trim        # cut video into cuts (10s each)
			 --audio       # extract wav given the video list
			 --frame       # extract frames given the video list
			 --at          # extract direction audio, see "at_utils/main.sh"
			 --pack        # downscale the extract frames folder and pack into a frames.npy

			 --force       # originally the script will skip the existing files, use this to force re-writing all files
			 --debug       # show debug information
			 --verbose     # make the processing visible
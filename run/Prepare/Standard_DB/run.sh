



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
# Data Preparation

### Step 1 Anonymous

```
# Follow the instruction in the file
jupyter notebook   Anonymous/face_anomolyze.ipynb
```



### Step 2 Standard

```
%cd Standard
python extract_data.py  + the following action you want
			 --trim        # cut video into cuts (10s each)
			 --audio       # extract wav given the video list
			 --frame       # extract frames given the video list
			 --at          # extract direction audio, see "at_utils/main.sh"
			 --pack        # downscale the extract frames folder and pack into a frames.npy
			 
			 --force       # originally the script will skip the existing files, use this to force re-writing all files
			 --debug       # show debug information
			 --verbose     # make the processing visible
```




### Step 3 Features

```
%cd Features
python extract_feat.py  + the following action you want
			 --audio       # Extract audio feature, write "audio_feat.npy"
			 --video       # Extract video feature, write "video_feat.npy"
			 --MAE         # Extract video feature using MAE pre-training model

			 --force       # originally the script will skip the existing files, use this to force re-writing all files
			 --debug       # show debug information
			 --verbose     # make the processing visible
			 
# Or run
bash video_I3D_prepare.py

```


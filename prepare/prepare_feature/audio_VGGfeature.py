import argparse
import os, glob
from numpy import genfromtxt
from pydub import AudioSegment
import numpy as np
import torchaudio
import tensorflow as tf
import tensorflow_hub as hub
import torch
import cv2


sr = 16000
snippet_size = int(sr * 1.2)  # sr rate = 22050




vggmodel = hub.load('https://tfhub.dev/google/vggish/1')



# Snippet size (In terms of no. of frames).


# Returns feature sequence from audio 'filename'
def getFeature(filename):
	# audios / *

	feat_file = filename.replace(".wav", "_feat.csv")
	if os.path.isfile(feat_file):
		print("skipping", filename)
		return

	video_path = filename.replace("audios/", "").replace('.wav', ".mp4")


	vid = cv2.VideoCapture(video_path)
	frameCnt = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
	print("video:", video_path, "frameCnt:", frameCnt)


	filename = filename.split('.')[0]

	# Initialize Feature Vector
	featureVec = tf.Variable([[ 0 for i in range(128) ]], dtype='float32')

	# Load audio file as tensor
	audio, sr = torchaudio.load(filename + '.wav')
	# Convert to mono
	audio = audio.mean(axis=0)
	# Resample to 16kHz
	audio = torchaudio.transforms.Resample(sr, 16000)(audio.view(1,-1))[0]

	# Iterate over all snippets and extract feature vector
	pointerr = len(audio) // frameCnt
	frameSize = len(audio) // frameCnt

	for i in range(frameCnt):
		# Get audio segment b/w start_time and end_time
		chunk = audio[max(0, pointerr - (snippet_size // 2)):min(len(audio), pointerr + (snippet_size // 2))]
		if len(chunk) < snippet_size:
			chunk = torch.from_numpy(np.pad(chunk, pad_width=(0, snippet_size - len(chunk)), mode='constant', constant_values=0))

		# Extract feature vector sequence
		feature = vggmodel(chunk)
		# Combine vector sequences by taking mean to represent whole segment. (i.e. convert (Ax128) -> (1x128))
		if len(feature.shape) == 2:
			feature = tf.reduce_mean(feature, axis=0)
		# Concatenate to temporal feature vector sequence
		featureVec = tf.concat([featureVec, [feature]], 0)
		pointerr += frameSize

	# Removing first row with zeroes
	featureVec = featureVec[1:].numpy()

	# Save as csv
	np.savetxt(feat_file, featureVec, delimiter=",", header=','.join([ 'f' + str(i) for i in range(featureVec.shape[1]) ]))
	# print(filename + ' Done')



# -------- files ----------
from glob import glob

root = '/bask/projects/j/jiaoj-3d-vision/360XProject/Data'


_360_list = glob(os.path.join(root, "Inside", "*", "*", "360/360_panoramic_cut/audios/*")) + \
          glob(os.path.join(root, "Outside", "*", "*", "360/360_panoramic_cut/audios/*"))


_360_front_list = glob(os.path.join(root, "Inside", "*", "*", "360/front_view_cut/audios/*")) + \
          glob(os.path.join(root, "Outside", "*", "*", "360/front_view_cut/audios/*"))


_Snapchat_list = glob(os.path.join(root, "Inside", "*", "*", "Snapchat/*/binocular_cut/audios/*")) + \
          glob(os.path.join(root, "Outside", "*", "*", "Snapchat/*/binocular_cut/audios/*"))


from tqdm import tqdm

def get_feature(folder):
	for file in tqdm(folder):
		if file.endswith('.wav'):
			getFeature(file)

get_feature(_360_list)
get_feature(_360_front_list)
get_feature(_Snapchat_list)


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

# import torch_tensorrt


sr = 16000
snippet_size = int(sr * 1.2)  # sr rate = 22050


vggmodel = hub.load('https://tfhub.dev/google/vggish/1')

# trt_model_fp16 = torch_tensorrt.compile(vggmodel, inputs = [torch_tensorrt.Input((128, 3, 224, 224), dtype=torch.half)],
#     enabled_precisions = {torch.half}, # Run with FP16
#     workspace_size = 1 << 22
# )
#
# import torch.onnx
# torch.onnx.export(resnext50_32x4d, dummy_input,
#                   "resnet50_onnx_model.onnx", verbose=False)
#
# # ./trtexec --onnx=resnet50_onnx_model.onnx --saveEngine=resnet_engine.trt



# Returns feature sequence from audio 'filename'
def getFeature(filename, force=False):
	# audios / *


	feat_file = filename.replace("audio/", "audio_feat/").replace(".wav", ".npy")
	video_path = filename.replace("audio/", "video/").replace(".wav", ".mp4")

	if os.path.exists(feat_file) and not force:
		print("skipping", filename)
		return


	vid = cv2.VideoCapture(video_path)
	frameCnt = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
	if frameCnt<0.5:
		print("Frame == 0:", filename)
		return


	# Initialize Feature Vector
	featureVec = tf.Variable([[0]*128], dtype='float32')

	# Load audio file as tensor
	audio, sr = torchaudio.load(filename)
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

		# print("chunk:", chunk.shape)   # chunk: torch.Size([19200])

		with torch.no_grad():
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

	print("featureVec:", featureVec.shape)   # (frame_Cnt, 128)

	# Save as csv
	np.save(feat_file, featureVec)


# -------- files ----------
from glob import glob



video_paths = '../360x/360data/360x_feat'


video_list = glob(os.path.join(video_paths, "audio/360", "*"))


os.makedirs("../360x/360data/360x_feat/audio_feat/360", exist_ok=True)


video_list.sort(reverse=False)   # reverse=True|False

from random import shuffle

shuffle(video_list)

print(f"len of {len(video_list)}")

from tqdm import tqdm

def get_feature(folder):
	for file in tqdm(folder):
		if file.endswith('.wav'):
			getFeature(file)

get_feature(video_list)



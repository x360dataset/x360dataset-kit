import numpy as np
import torch
import sys, os
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from .models.extract_vggish import ExtractVGGish

extractor = ExtractVGGish()
set_sr = 16000
snippet_size = int(set_sr * 1.2)  # sr rate = 22050

@torch.no_grad()
def get_AudioFeature(files, force=False, pretrain_path = None):
	# audios / *
	
	if pretrain_path:
		state_dict = torch.load(pretrain_path)
		extractor.name2module['model'].load_state_dict(state_dict)
  
	for videofile in tqdm(files):

		audiofile = videofile.replace('.mp4', '.wav')
		mp4name = videofile.split('/')[-1]
		featfile = videofile.replace(mp4name, "audio_feat.npy")

		if os.path.exists(featfile) and not force:
			print("=== Already exist:", featfile)
			continue

		with autocast(enabled=True):
			featureVec = extractor.extract(videofile)

		if isinstance(featureVec, type(None)):
			print("=== Error in extracting: ", videofile)
			continue

		np.save(featfile, featureVec)
		print("=== Save to: ", featfile, " with shape: ", featureVec.shape)


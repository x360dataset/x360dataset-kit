import os
import pathlib
from typing import Dict

import numpy as np
import torch
from ._base.base_extractor import BaseExtractor
from .vggish.vggish_src.vggish_slim import VGGish
from .utils.utils import extract_wav_from_mp4


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ExtractVGGish(BaseExtractor):

    def __init__(self) -> None:
        # init the BaseExtractor
        super().__init__(
            feature_type=None,
            on_extraction=None,
            tmp_path=None,
            output_path=None,
            keep_tmp_files=None,
            device=device,
        )
        # (Re-)Define arguments for this class

        self.output_feat_keys = [self.feature_type]
        self.name2module = self.load_model()

    @torch.no_grad()
    def extract(self, video_path: str) -> Dict[str, np.ndarray]:
        """Extracts features for a given video path.

        Arguments:
            video_path (str): a video path from which to extract features

        Returns:
            Dict[str, np.ndarray]: feature name (e.g. 'fps' or feature_type) to the feature tensor
        """
        file_ext = pathlib.Path(video_path).suffix

        if file_ext == '.mp4':
            # extract audio files from .mp4
            audio_wav_path = video_path.replace('.mp4', '.wav')
        elif file_ext == '.wav':
            audio_wav_path = video_path

        else:
            raise NotImplementedError

        audio_aac_path = None
        with torch.no_grad():
            vggish_stack = self.name2module['model'](audio_wav_path, self.device)

        if isinstance(vggish_stack, type(None)):
            return None

        print("vggish_stack:", vggish_stack.shape)

        # removes the folder with audio files created during the process



        return vggish_stack.cpu().numpy()

    def load_model(self) -> torch.nn.Module:
        """Defines the models, loads checkpoints, sends them to the device.


        Returns:
            {Dict[str, torch.nn.Module]}: model-agnostic dict holding modules for extraction
        """
        model = VGGish()
        model = model.to(self.device)
        model.eval()
        return {'model': model}

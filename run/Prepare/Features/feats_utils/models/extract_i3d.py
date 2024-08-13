import os
from typing import Dict

import cv2
import numpy as np
import torch
import torchvision

# import sys
# sys.path.append("./feat_utils")


from ._base.base_extractor import BaseExtractor
from .i3d.i3d_src.i3d_net import I3D


from .transforms import (Clamp, PermuteAndUnsqueeze, PILToTensor,
                               ResizeImproved, ScaleTo1_1, TensorCenterCrop, Resize,
                               ToFloat, ToUInt8)

from .utils.io import reencode_video_with_diff_fps
from .utils.utils import dp_state_to_normal, show_predictions_on_dataset



class MyTensorCenterCrop(object):

    def __init__(self, crop_size: int) -> None:
        self.crop_size = crop_size

    def __call__(self, tensor: torch.FloatTensor) -> torch.FloatTensor:
        H, W = tensor.size(-2), tensor.size(-1)
        from_H = ((H - self.crop_size) // 2)
        from_W = ((W - self.crop_size*2) // 2)
        to_H = from_H + self.crop_size
        to_W = from_W + self.crop_size*2


        return tensor[..., from_H:to_H, from_W:to_W]





class ExtractI3D(BaseExtractor):

    def __init__(self, args) -> None:
        # init the BaseExtractor
        super().__init__(
            feature_type=args.feature_type,
            on_extraction=args.on_extraction,
            tmp_path=args.tmp_path,
            output_path=args.output_path,
            keep_tmp_files=args.keep_tmp_files,
            device=args.device,
        )
        # (Re-)Define arguments for this class
        
        self.streams = ['rgb']       #, 'flow'] if args.streams is None else [args.streams]
        self.flow_type = args.flow_type
        self.i3d_classes_num = 400
        self.min_side_size = 256
        self.central_crop_size = 224

        MODEL_PATH = "/bask/projects/j/jiaoj-3d-vision/Hao/360x/360x_Video_Experiments/exp/SSL-CLS"  \
                    "/results/i3d_Modality_11111_CLS_clip1_maxsr1_ci16_frame16_11140744/epoch-13-acc-0.80625.pth.tar"

        self.weights = torch.load(MODEL_PATH)

        self.extraction_fps = args.extraction_fps
        self.step_size = 64 if args.step_size is None else args.step_size
        self.stack_size = 64 if args.stack_size is None else args.stack_size

        self.resize_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            ResizeImproved(self.min_side_size),
            PILToTensor(),
            ToFloat(),
        ])
        self.i3d_transforms = {
            'rgb': torchvision.transforms.Compose([
                # Resize((self.central_crop_size, 2*self.central_crop_size)),
                MyTensorCenterCrop(self.central_crop_size),   # My
                ScaleTo1_1(),
                PermuteAndUnsqueeze()
            ]),
            'flow': torchvision.transforms.Compose([
                TensorCenterCrop(self.central_crop_size),
                Clamp(-20, 20),
                ToUInt8(),
                ScaleTo1_1(),
                PermuteAndUnsqueeze()
            ])
        }
        self.show_pred = args.show_pred
        self.output_feat_keys = self.streams #+ ['fps', 'timestamps_ms']
        self.name2module = self.load_model()

    @torch.no_grad()
    def extract(self, npy) -> Dict[str, np.ndarray]:
        """The extraction call. Made to clean the forward call a bit.

        Arguments:
            video_path (str): a video path from which to extract features

        Returns:
            Dict[str, np.ndarray]: feature name (e.g. 'fps' or feature_type) to the feature tensor
        """

        # take the video, change fps and save to the tmp folder

        # timestamp when the last frame in the stack begins (when the old frame of the last pair ends)
        timestamps_ms = []
        rgb_stack = []
        feats_dict = {stream: [] for stream in self.streams}

        # sometimes when the target fps is 1 or 2, the first frame of the reencoded video is missing
        # and cap.read returns None but the rest of the frames are ok. timestep is 0.0 for the 2nd frame in
        # this case
        first_frame = True
        padder = None
        stack_counter = 0
        for rgb in npy:

            H, W, _ = rgb.shape
            ratio = H/240
            H = int(H/ratio)
            W = int(W/ratio)


            # rgb: (256, 512, 3)
            rgb = cv2.resize(rgb, (W, H))

            # preprocess the image
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = self.resize_transforms(rgb)
            rgb = rgb.unsqueeze(0)

            # print("rgb:", rgb.shape)   #  1, 3, 256, 512

            rgb_stack.append(rgb)

            # - 1 is used because we need B+1 frames to calculate B frames
            if len(rgb_stack) - 1 == self.stack_size:
                batch_feats_dict = self.run_on_a_stack(rgb_stack, stack_counter, padder)
                for stream in self.streams:
                    feats_dict[stream].extend(batch_feats_dict[stream].tolist())


                # leaving the elements if step_size < stack_size so they will not be loaded again
                # if step_size == stack_size one element is left because the flow between the last element
                # in the prev list and the first element in the current list
                rgb_stack = rgb_stack[self.step_size:]
                stack_counter += 1
                # timestamps_ms.append(cap.get(cv2.CAP_PROP_POS_MSEC))


        # removes the video with different fps if it was created to preserve disk space
        if (self.extraction_fps is not None) and (not self.keep_tmp_files):
            os.remove(video_path)

        # transforms list of features into a np array
        feats_dict = {stream: np.array(feats) for stream, feats in feats_dict.items()}

        return feats_dict

    def run_on_a_stack(self, rgb_stack, stack_counter, padder=None) -> Dict[str, torch.Tensor]:
        models = self.name2module['model']
        flow_xtr_model = self.name2module.get('flow_xtr_model', None)
        rgb_stack = torch.cat(rgb_stack).to(self.device)

        batch_feats_dict = {}
        for stream in self.streams:
            # if i3d stream is flow, we first need to calculate optical flow, otherwise, we use rgb
            # `end_idx-1` and `start_idx+1` because flow is calculated between f and f+1 frames
            # we also use `end_idx-1` for stream == 'rgb' case: just to make sure the feature length
            # is same regardless of whether only rgb is used or flow
            if stream == 'flow':
                if self.flow_type == 'raft':
                    stream_slice = flow_xtr_model(padder.pad(rgb_stack)[:-1], padder.pad(rgb_stack)[1:])
                elif self.flow_type == 'pwc':
                    stream_slice = flow_xtr_model(rgb_stack[:-1], rgb_stack[1:])
                else:
                    raise NotImplementedError
            elif stream == 'rgb':
                stream_slice = rgb_stack[:-1]
            else:
                raise NotImplementedError
            # apply transforms depending on the stream (flow or rgb)
            stream_slice = self.i3d_transforms[stream](stream_slice)
            # print("stream_slice:", stream_slice.shape)  # 1, 3, 16, 224, 224

            # extract features for a stream
            batch_feats_dict[stream] = models[stream](stream_slice, features=True)  # (B, 1024)
            # add features to the output dict
            self.maybe_show_pred(stream_slice, self.name2module['model'][stream], stack_counter)

        return batch_feats_dict

    def load_model(self) -> Dict[str, torch.nn.Module]:
        """Defines the models, loads checkpoints, sends them to the device.
        Since I3D is two-stream, it may load a optical flow extraction model as well.

        Returns:
            Dict[str, torch.nn.Module]: model-agnostic dict holding modules for extraction and show_pred
        """
        # flow_model_paths = {'pwc': DATASET_to_PWC_CKPT_PATHS['sintel'], 'raft': DATASET_to_RAFT_CKPT_PATHS['sintel']}
       
        i3d_weights_paths = {
            'rgb': './feats_utils/models/i3d/checkpoints/i3d_rgb.pt',
            'flow': './feats_utils/models/i3d/checkpoints/i3d_flow.pt',
        }
        name2module = {}

        # if "flow" in self.streams:
        #     # Flow extraction module
        #     if self.flow_type == 'pwc':
        #         from models.pwc.pwc_src.pwc_net import PWCNet
        #         flow_xtr_model = PWCNet()
        #     elif self.flow_type == 'raft':
        #         flow_xtr_model = RAFT()
        #     # Preprocess state dict
        #     state_dict = torch.load(flow_model_paths[self.flow_type], map_location='cpu')
        #     state_dict = dp_state_to_normal(state_dict)
        #     flow_xtr_model.load_state_dict(state_dict)
        #     flow_xtr_model = flow_xtr_model.to(self.device)
        #     flow_xtr_model.eval()
        #     name2module['flow_xtr_model'] = flow_xtr_model

        # Feature extraction models (rgb and flow streams)
        i3d_stream_models = {}
        for stream in self.streams:
            i3d_stream_model = I3D(num_classes=self.i3d_classes_num, modality=stream)
            i3d_stream_model.load_state_dict(torch.load(i3d_weights_paths[stream], map_location='cpu'))
            i3d_stream_model = i3d_stream_model.to(self.device)
            i3d_stream_model.eval()
            i3d_stream_models[stream] = i3d_stream_model
        name2module['model'] = i3d_stream_models

        return name2module

    def maybe_show_pred(self, stream_slice: torch.Tensor, model: torch.nn.Module, stack_counter: int) -> None:
        if self.show_pred:
            softmaxes, logits = model(stream_slice, features=False)
            print(f'At stack {stack_counter} ({model.modality} stream)')
            show_predictions_on_dataset(logits, 'kinetics')

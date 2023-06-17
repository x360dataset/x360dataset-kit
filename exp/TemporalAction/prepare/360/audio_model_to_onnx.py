import os

import torch
import tensorflow_hub as hub





vggmodel = hub.load('https://tfhub.dev/google/vggish/1')

os.makedirs("onnx", exist_ok=True)

dummy_input = torch.randn(1, 3, 300, 300, device='cuda')



torch.onnx.export(vggmodel, dummy_input, "onnx/vggish.onnx" , verbose=False)




import numpy as np

USE_FP16 = True
target_dtype = np.float16 if USE_FP16 else np.float32
BATCH_SIZE = 1


# to TensorRT
import tensorrt

if USE_FP16:
    os.system('trtexec --onnx=onnx/vggish.onnx --saveEngine=onnx/vggish.trt  --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16')
else:
    os.system('trtexec --onnx=onnx/vggish.onnx --saveEngine=onnx/vggish.trt  --explicitBatch')



from __future__ import print_function

import os
import argparse
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import torchvision.transforms as transforms
import common
from PIL import Image
from lib.utils.augmentations import letterbox_for_img
import torch
from multiple import *
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

EXPLICIT_BATCH = []
if trt.__version__[0] >= '7':
    EXPLICIT_BATCH.append(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

def load_img(path):
    img0 = cv2.imread(path, cv2.IMREAD_COLOR |
                      cv2.IMREAD_IGNORE_ORIENTATION)  # BGR
    h0, w0 = img0.shape[:2]

    img, ratio, pad = letterbox_for_img(img0.copy(), new_shape=640, auto=True)
    h, w = img.shape[:2]
    shapes = (h0, w0), ((h / h0, w / w0), pad)
    img = np.ascontiguousarray(img)
    return img, img0, shapes
def load_engine(trt_file_path, verbose=False):
    """Build a TensorRT engine from a TRT file."""
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
    print('Loading TRT file from path {}...'.format(trt_file_path))
    with open(trt_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine
def quantize(a):
    maxa,mina=np.max(a),np.min(a)
    c = (maxa- mina)/(255)
    d = np.round_((mina*255)/(maxa - mina))
    a = np.round_((1/c)*a-d)
    return a.astype('uint8'), c, d
engine = load_engine('jetson-trt/seg.trt', False)
def inference_seg(out2):
    '''
    input: outs[2] of backbone
    output: 1 array 
    '''
    with engine.create_execution_context() as context:
        h_inputs, h_outputs, bindings, stream = common.allocate_buffers(engine)
        h_inputs[0].host = out2
        trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=h_inputs, outputs=h_outputs, stream=stream)
        trt_outputs[0] = np.reshape(trt_outputs[0],(1,2,384,640))
    return trt_outputs[0]

def main():
    img, img_det, shapes = load_img(
        '/home/tx2/YOLOP/inference/images/0ace96c3-48481887.jpg')
    img = transform(img)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    out2 = np.load('outNe.npy')
    #truyen out 2 vao
    out_seg = inference_seg(out2)
    
    print(type(img), img.dtype, shapes)

    color_area = post_process_seg( torch.tensor(out_seg), img, shapes)
    cv2.imwrite('test_trt2_seg_a.jpg', color_area)
if __name__ =='__main__':
    main()
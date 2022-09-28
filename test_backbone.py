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
    # img0 = cv2.imread(path, cv2.IMREAD_COLOR |
    #                   cv2.IMREAD_IGNORE_ORIENTATION)  # BGR

    _, img0 = path.read(cv2.IMREAD_COLOR |
                      cv2.IMREAD_IGNORE_ORIENTATION)
    h0, w0 = img0.shape[:2]

    img = cv2.resize(img0, (640, 384))

    # img, ratio, pad = letterbox_for_img(img0.copy(), new_shape=640, auto=True)
    # h, w = img.shape[:2]
    # shapes = (h0, w0), ((h / h0, w / w0), pad)
    # print(img.shape)
    img = np.ascontiguousarray(img)
    return img, img0, _
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

engine = load_engine('jetson-trt/bb.trt', False)

h_inputs, h_outputs, bindings, stream = common.allocate_buffers(engine)
def inference_bb(img):
    '''
    input: image as array
    output: 3 tensor
    '''
    with engine.create_execution_context() as context:
        h_inputs[0].host = img
        trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=h_inputs, outputs=h_outputs, stream=stream)
        trt_outputs[0] = trt_outputs[0].reshape(1, 256, 12, 20)
        trt_outputs[1] = trt_outputs[1].reshape(1, 128, 24, 40)
        trt_outputs[2] = trt_outputs[2].reshape(1, 256, 48, 80)
    return trt_outputs

# with open('dumm.txt') as context:
def main():
    # img_original, img_det, shapes = load_img(
    #     '/home/tx2/YOLOP/inference/images/0ace96c3-48481887.jpg')
    
    cap = cv2.VideoCapture('/home/tx2/AV-multi-machine/YOLOP/inference/videos/1.mp4')

    pre = 0

    while True:
        pre = time.time()
        img_original, img_det, shapes = load_img(cap)
        img = transform(img_original)
        img = img.float()
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        outs = inference_bb(img.numpy())
        print(time.time() - pre)
if __name__ =='__main__':
    main()
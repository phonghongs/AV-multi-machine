from __future__ import print_function

import os
import argparse
from re import S
import cv2
import tensorrt as trt
import pycuda.driver as cuda
# import pycuda.autoinit
import numpy as np
import common
from PIL import Image
from lib.utils.augmentations import letterbox_for_img
import torch
from multiple import *
import threading
from queue import Empty, Queue


EXPLICIT_BATCH = []
if trt.__version__[0] >= '7':
    EXPLICIT_BATCH.append(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

global pre
pre = 0





class Inference(threading.Thread):
    def __init__(self, args=(), kwargs=None):
        threading.Thread.__init__(self, args=(), kwargs=None)

        cuda.init()
        device = cuda.Device(0) # enter your gpu id here
        self.cfx = device.make_context()
        self.queueIn = args[0]
        self.queueOut = args[1]
        self.daemon = True
        self.quit = False
        self.engine = self.load_engine(args[2], False)
        self.h_inputs, self.h_outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)
        # self.h_inputs = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(0)), dtype=np.float32)
        # self.h_outputs = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(1)), dtype=np.float32)
        # # Allocate device memory for inputs and outputs.
        # self.d_input = cuda.mem_alloc(self.h_inputs.nbytes)
        # self.d_output = cuda.mem_alloc(self.h_outputs.nbytes)
        # # Create a stream in which to copy inputs/outputs and run inference.
        # self.stream = cuda.Stream()

    def load_engine(self, trt_file_path, verbose=False):
        """Build a TensorRT engine from a TRT file."""
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
        print('Loading TRT file from path {}...'.format(trt_file_path))
        with open(trt_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    def inference_bb(self, img):
        '''
        input: image as array
        output: 3 tensor
        '''
        with self.engine.create_execution_context() as context:
            self.h_inputs[0].host = img
            trt_outputs = common.do_inference_v2(
                context, 
                bindings=self.bindings, 
                inputs=self.h_inputs, 
                outputs=self.h_outputs, 
                stream=self.stream
            )
            trt_outputs[0] = trt_outputs[0].reshape(1, 256, 12, 20)
            trt_outputs[1] = trt_outputs[1].reshape(1, 128, 24, 40)
            trt_outputs[2] = trt_outputs[2].reshape(1, 256, 48, 80)
        return trt_outputs

    def run(self):
        print(threading.currentThread().getName())

        while not self.quit:
            # if self.queueIn.empty() == False:
            val = self.queueIn.get()
            outs = self.inference_bb(val.numpy())
            print("[Total time] ", time.time() - pre, val.numpy().shape)
        self.cfx.pop()
        del self.cfx
    
    def __del__(self):
        del self.engine


def load_img(path):
    img0 = cv2.imread(path, cv2.IMREAD_COLOR |
                      cv2.IMREAD_IGNORE_ORIENTATION)  # BGR
    h0, w0 = img0.shape[:2]

    img, ratio, pad = letterbox_for_img(img0.copy(), new_shape=640, auto=True)
    h, w = img.shape[:2]
    shapes = (h0, w0), ((h / h0, w / w0), pad)
    img = np.ascontiguousarray(img)
    return img, img0, shapes

def quantize(a):
    maxa,mina=np.max(a),np.min(a)
    c = (maxa- mina)/(255)
    d = np.round_((mina*255)/(maxa - mina))
    a = np.round_((1/c)*a-d)
    return a.astype('uint8'), c, d

def load_engine(trt_file_path, verbose=False):
    """Build a TensorRT engine from a TRT file."""
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
    print('Loading TRT file from path {}...'.format(trt_file_path))
    with open(trt_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

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
    img_original, img_det, shapes = load_img(
        '/home/tx2/YOLOP/inference/images/0ace96c3-48481887.jpg')
    
    pre = 0

    while True:
        pre = time.time()
        img = transform(img_original)
        img = img.float()
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        outs = inference_bb(img.numpy())
        print(time.time() - pre)

def parallelism_main():
    img_original, img_det, shapes = load_img(
        '/home/tx2/YOLOP/inference/images/0ace96c3-48481887.jpg')

    threads = []
    q = Queue()

    threads.append(Transform(args=(q, img_original)))
    threads[0].start()
    new_pre = time.time()
    while time.time() - new_pre < 30:
        preee = time.time()
        val = q.get()
        outs = inference_bb(val.numpy())
        print("[Total time] ", time.time() - preee, val.numpy().shape)
    
    threads[0].quit = True
    threads[0].join()


if __name__ =='__main__':
    parallelism_main()

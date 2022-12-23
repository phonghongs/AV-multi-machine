import threading
import time
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import common
from Script.Component.ThreadDataComp import ThreadDataComp
from multiple import *

class InferenceSegment():
    def __init__(self, _threadDataComp: ThreadDataComp):
        self.threadDataComp = _threadDataComp
        self.engine = self.load_engine(self.threadDataComp.ModelPath, False)
        self.h_inputs, self.h_outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)

    def load_engine(self, trt_file_path, verbose=False):
        """Build a TensorRT engine from a TRT file."""
        TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
        print('Loading TRT file from path {}...'.format(trt_file_path))
        with open(trt_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
        return engine

    def inference_seg(self, imgTensor):
        '''
        input: image as array
        output: 3 tensor
        '''
        with self.engine.create_execution_context() as context:
            h_inputs, h_outputs, bindings, stream = common.allocate_buffers(self.engine)
            h_inputs[0].host = imgTensor
            trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=h_inputs, outputs=h_outputs, stream=stream)
            trt_outputs[0] = np.reshape(trt_outputs[0],(1,2,384,640))
        return trt_outputs[0]

    def run(self):
        print(threading.currentThread().getName())

        timecount = 0.00001
        totalTime = 0
        firstTime = True
        while not self.threadDataComp.isQuit:
            pre = time.time()
            output = self.threadDataComp.ImageQueue.get()
            if (firstTime):
                pre = time.time()
                firstTime = False
            if output is None:
                print("[InferenceSeg] Error when get Image in queue")
                break
            
            [getTensor, timestamp] = output

            try:
                outs = self.inference_seg(getTensor)
                self.threadDataComp.TransformQueue.put([outs, timestamp])
                # print("[InferenceSeg]: ", time.time() - pre)
                timecount += 1
                totalTime += time.time() - pre
            except Exception as e:
                print("[InferenceSeg]: Error when inference Segment : ", e)
        print("[InferenceSeg]: Total Time : ", totalTime/timecount)

    def delInstance(self):
        del self.engine

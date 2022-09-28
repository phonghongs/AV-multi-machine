import threading
import time
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import common
from Script.Component.ThreadDataComp import ThreadDataComp


class Inference(threading.Thread):
    def __init__(self, _threadDataComp: ThreadDataComp):
        threading.Thread.__init__(self, args=(), kwargs=None)
        self.daemon = True
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

        while not self.threadDataComp.isQuit:
            pre = time.time()

            with self.threadDataComp.TransformCondition:
                self.threadDataComp.TransformCondition.wait()
            getImage = self.threadDataComp.TransformQueue.get(timeout=1)

            if getImage is None:
                print("[Inference] Error when get Image in queue")
                break
            
            outs = self.inference_bb(getImage.numpy())
            
            self.threadDataComp.OutputQueue.put(
                outs
            )
            self.threadDataComp.totalTime.put(time.time() - pre)
            print("[Inference] Total Time", time.time() - pre)
    
    def __del__(self):
        del self.engine

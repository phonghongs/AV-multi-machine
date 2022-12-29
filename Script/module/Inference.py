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

class Inference():
    def __init__(self, _threadDataComp: ThreadDataComp):
        self.daemon = True
        self.threadDataComp = _threadDataComp
        self.engine = self.load_engine(self.threadDataComp.ModelPath, False)
        self.h_inputs, self.h_outputs, self.bindings, self.stream = common.allocate_buffers(self.engine)

        self.img, img_det, self.shapes = load_img(
        '/home/tx2/YOLOP/inference/images/0ace96c3-48481887.jpg')
        self.img = transform(self.img)
        if self.img.ndimension() == 3:
            self.img = self.img.unsqueeze(0)

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
            
            # trt_outputs[0] = (trt_outputs[0].reshape(1, 256, 12, 20) * 255).round().astype(np.uint8)
            # trt_outputs[1] = (trt_outputs[1].reshape(1, 128, 24, 40) * 255).round().astype(np.uint8)
            # trt_outputs[2] = (trt_outputs[2].reshape(1, 256, 48, 80) * 255).round().astype(np.uint8)

            trt_outputs[0] = trt_outputs[0].reshape(1, 256, 12, 20)
            trt_outputs[1] = trt_outputs[1].reshape(1, 128, 24, 40)
            trt_outputs[2] = trt_outputs[2].reshape(1, 256, 48, 80)
        return trt_outputs

    def run(self):
        print(threading.currentThread().getName())
        timecount = 0.00001
        totalTime = 0
        while not self.threadDataComp.isQuit:
            pre = time.time()

            # with self.threadDataComp.TransformCondition:
            #     self.threadDataComp.TransformCondition.wait()
            output = self.threadDataComp.TransformQueue.get()
            if output is None:
                print("[Inference] Error when get Image in queue")
                break
            
            [getImage, timestamp] = output
            outs = self.inference_bb(getImage.numpy())

            # color_area = post_process_seg( torch.tensor(outs[2]), self.img, self.shapes)

            # with self.threadDataComp.OutputCondition:
            #     self.threadDataComp.output = outs

            # if self.threadDataComp.QuantaQueue.full():
            #     self.threadDataComp.QuantaQueue.get()
            self.threadDataComp.QuantaQueue.put([outs, timestamp])

            # with self.threadDataComp.QuantaCondition:
            #     if self.threadDataComp.QuantaQueue.qsize() > 0:
            #         self.threadDataComp.QuantaCondition.notifyAll()

            timecount += 1
            totalTime += time.time() - pre
        print("[Inference]: Total Time : ", totalTime/timecount)

    def __del__(self):
        del self.engine


normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)

transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])


def load_img(path):
    img0 = cv2.imread(path, cv2.IMREAD_COLOR |
                      cv2.IMREAD_IGNORE_ORIENTATION)  # BGR
    h0, w0 = img0.shape[:2]

    img, ratio, pad = letterbox_for_img(img0.copy(), new_shape=640, auto=True)
    h, w = img.shape[:2]
    shapes = (h0, w0), ((h / h0, w / w0), pad)
    img = np.ascontiguousarray(img)
    return img, img0, shapes


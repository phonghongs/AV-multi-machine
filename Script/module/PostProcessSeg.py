import time
import cv2
import threading
import numpy as np
import torch
from torch import Tensor
import torchvision.transforms as transforms
from Script.Component.ThreadDataComp import ThreadDataComp

class PostProcessSeg(threading.Thread):
    def __init__(self,  _threadDataComp: ThreadDataComp):
        threading.Thread.__init__(self, args=(), kwargs=None)
        self.threadDataComp = _threadDataComp
        self.daemon = True
        self.output = cv2.VideoWriter('out1.avi', 
                                cv2.VideoWriter_fourcc(*'MJPG'),
                                10, (640, 360))

    def run(self):
        print(threading.currentThread().getName())

        timecount = 0.00001
        totalTime = 0

        while not self.threadDataComp.isQuit:
            getSeg = self.threadDataComp.TransformQueue.get()
            pre = time.time()
            if getSeg is None:
                print("[TransFromImage] Error when get Image in queue")
                break

            try:
                getSeg = torch.tensor(getSeg)
                # shapes = ((720, 1280), ((0.5333333333333333, 0.5), (0.0, 12.0))) #720 1280
                shapes = ((360, 640), ((1.0666666666666667, 1.0), (0.0, 12.0)))
                height, width = 384, 640
                # h, w, _ = img_det.shape
                pad_w, pad_h = shapes[1][1]
                pad_w = int(pad_w)
                pad_h = int(pad_h)
                ratio = shapes[1][0][1]
                da_predict = getSeg[:, :, pad_h:(height-pad_h), pad_w:(width-pad_w)]
                da_seg_mask = torch.nn.functional.interpolate(
                    da_predict, scale_factor=int(1/ratio), mode='area')
                _, da_seg_mask = torch.max(da_seg_mask, 1)
                da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
                
                color_area = np.zeros(
                    (da_seg_mask.shape[0], da_seg_mask.shape[1], 3), dtype=np.uint8)
                color_area[da_seg_mask == 1] = [0, 255, 0]
                self.output.write(color_area)
                # self.threadDataComp.QuantaQueue.put(da_seg_mask)
                timecount += 1
                totalTime += time.time() - pre
            except Exception as e:
                print("[TransformImage] Error when transform : ", e)

        print("[PostProcessSeg]: Total Time : ", totalTime/timecount)

    def delInstance(self):
        self.output.release()
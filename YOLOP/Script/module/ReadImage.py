import threading
import time
import cv2
import logging
import numpy as np
from lib.utils.augmentations import letterbox_for_img
from Script.Component.ThreadDataComp import ThreadDataComp

class ReadImage(threading.Thread):
    def __init__(self, _threadDataComp: ThreadDataComp):
        threading.Thread.__init__(self, args=(), kwargs=None)
        self.threadDataComp = _threadDataComp
        self.videoCap = cv2.VideoCapture(self.threadDataComp.ImagePath)
        self.daemon = True

    def run(self):
        print(threading.currentThread().getName())
        while not self.threadDataComp.isQuit:
            pre = time.time()
            result = []
            try:
                result = self.load_img()
            except Exception as ex:
                print("[ReadImage]: ", ex)
                self.threadDataComp.isQuit = True
                break

            if self.threadDataComp.ImageQueue.full():
                self.threadDataComp.ImageQueue.get()
            self.threadDataComp.ImageQueue.put(result)

            with self.threadDataComp.ImageCondition:
                if self.threadDataComp.ImageQueue.qsize() > 0:
                    self.threadDataComp.ImageCondition.notifyAll()

            self.threadDataComp.totalTime.put(time.time() - pre)
            # print("[ReadImage] Done at: ", time.time() - pre)
            # logging.debug("Done at: %s", time.time() - pre)

    def load_img(self):
        # img0 = cap.read(cv2.IMREAD_COLOR |
        #                 cv2.IMREAD_IGNORE_ORIENTATION)
        ret, img0 = self.videoCap.read(cv2.IMREAD_COLOR |
                            cv2.IMREAD_IGNORE_ORIENTATION)
        # img = cv2.resize(img0, (640, 384))
        h0, w0 = img0.shape[:2]

        img, ratio, pad = letterbox_for_img(img0.copy(), new_shape=640, auto=True)
        h, w = img.shape[:2]
        shapes = (h0, w0), ((h / h0, w / w0), pad)
        img = np.ascontiguousarray(img)
        return [img, img0, shapes]

import time
import threading
import numpy as np
import torchvision.transforms as transforms
from Script.Component.ThreadDataComp import ThreadDataComp

class Quanta(threading.Thread):
    def __init__(self,  _threadDataComp: ThreadDataComp):
        threading.Thread.__init__(self, args=(), kwargs=None)
        self.threadDataComp = _threadDataComp
        self.daemon = True

    def quantize(self, a):
        pre_a = time.time()
        # maxa,mina=np.max(a),np.min(a)
        maxa, mina = 8.0, -0.375
        # print('maxmin aaa: ', time.time()- pre_a, maxa, mina)
        
        # pre_a = time.time()
        c = (maxa- mina)/(255)
        # print('maxmin c: ', time.time()- pre_a)
        
        # pre_a = time.time()
        d = np.round_((mina*255)/(maxa - mina))
        # print('maxmin d: ', time.time()- pre_a)
        
        # pre_a = time.time()
        a = a/c - d
        # print('maxmin a: ', time.time()- pre_a)

        # print("______________________________________________")
        return a.astype('uint8')

    def run(self):
        print(threading.currentThread().getName())
        while not self.threadDataComp.isQuit:
            pre = time.time()

            # with self.threadDataComp.QuantaCondition:
            #     self.threadDataComp.QuantaCondition.wait()
            output = self.threadDataComp.QuantaQueue.get()

            if output is None:
                print("[TransFromImage] Error when get Image in queue")
                break

            # output[0] = (output[0] * 255).round().astype(np.uint8)
            # output[1] = (output[1] * 255).round().astype(np.uint8)
            # output[2] = (output[2] * 255).round().astype(np.uint8)

            # output[0] = self.quantize(output[0])
            # output[1] = self.quantize(output[1])
            output[2] = self.quantize(output[2])

            with self.threadDataComp.OutputCondition:
                self.threadDataComp.output = output

            self.threadDataComp.totalTime.put(time.time() - pre)

            print("[Quanta] Timer ", time.time() - pre)

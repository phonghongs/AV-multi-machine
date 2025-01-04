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
        self.f = -1000
        self.g = 1000

    # def quantize(self, a):
    #     pre_a = time.time()
    #     maxa,mina=np.max(a),np.min(a)
    #     # maxa, mina = 8.0, -0.375
    #     # print('maxmin aaa: ', time.time()- pre_a, maxa, mina)
    #     c = (maxa- mina)/(255)
    #     d = np.round_((mina*255)/(maxa - mina))
        
    #     a = a/c - d
    #     return a.astype('uint8')

    
    def quantize(self, a):
        # maxa,mina=np.max(a),np.min(a)
        maxa, mina = 9.7578125, -0.375
        c = (maxa- mina)/(255)
        d = np.round_((mina*255)/(maxa - mina))
        a = np.round_((1/c)*a-d)
        self.f = max(self.f, maxa)
        self.g = min(self.g, mina)
        # print(self.f, self.g)
        return a.astype('uint8')

    def run(self):
        print(threading.currentThread().getName())
        timecount = 0.00001
        totalTime = 0
        while not self.threadDataComp.isQuit:
            pre = time.time()

            # with self.threadDataComp.QuantaCondition:
            #     self.threadDataComp.QuantaCondition.wait()
            output = self.threadDataComp.QuantaQueue.get()

            if self.threadDataComp.isTimeProcess:
                pre = time.time()

            if output is None:
                print("[TransFromImage] Error when get Image in queue")
                break
            
            [outputTensor, timestamp] = output
            # output[0] = (output[0] * 255).round().astype(np.uint8)
            # output[1] = (output[1] * 255).round().astype(np.uint8)
            # output[2] = (output[2] * 255).round().astype(np.uint8)

            # output[0] = self.quantize(output[0])
            # output[1] = self.quantize(output[1])
            outputTensor[2] = self.quantize(outputTensor[2])
            # print(output[2].dtype, type(output[2]), output[2].shape)
            with self.threadDataComp.OutputCondition:
                self.threadDataComp.output = [outputTensor, timestamp]

            timecount += 1
            totalTime += time.time() - pre
        print("[Quanta]: Total Time : ", totalTime/timecount)

import logging
import time
import numpy as np
import threading
from Script.Component.ThreadDataComp import ThreadDataComp
from Script.module.ReadImage import ReadImage
from Script.module.TransfromImage import TransfromImage
from Script.module.Inference import Inference
from queue import Queue

global threadDataComp
threadDataComp = ThreadDataComp(
    Queue(maxsize=3), 
    Queue(maxsize=3), 
    Queue(), 
    threading.Condition(),
    threading.Condition(),
    threading.Condition(),    
    '/home/tx2/AV-multi-machine/YOLOP/inference/videos/data_test.mp4',
    'jetson-trt/bb.trt',
    False,
    Queue()
)

def main():
    # FORMAT = "%(asctime)-15s : %(levelname)s : %(threadname)s - %(message)s"
    # logging.basicConfig(filename="runtime.log", filemode='w', format=FORMAT)
    readImageTask = ReadImage(threadDataComp)
    tranformTask = TransfromImage(threadDataComp)
    inferenceTask = Inference(threadDataComp)

    readImageTask.start()
    tranformTask.start()
    inferenceTask.start()

    time.sleep(10)
    threadDataComp.isQuit = True
    while not threadDataComp.ImageQueue.empty():
        threadDataComp.ImageQueue.get()
    
    while not threadDataComp.TransformQueue.empty():
        threadDataComp.TransformQueue.get()
    

    readImageTask.join()
    tranformTask.join()
    inferenceTask.join()

    timeSize = threadDataComp.totalTime.qsize()
    count = 0
    while not threadDataComp.totalTime.empty():
        count += threadDataComp.totalTime.get()
    
    # out = threadDataComp.OutputQueue.get()
    # np.save('outNe.npy', out[2])
    print("[App]: ", threadDataComp.ImageQueue.qsize(), threadDataComp.TransformQueue.qsize(), threadDataComp.OutputQueue.qsize(), count / timeSize)

if __name__ == "__main__":
    main()

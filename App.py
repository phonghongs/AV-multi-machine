import logging
import time
import numpy as np
import threading
from Script.Component.ThreadDataComp import ThreadDataComp
from Script.Component.ConnectComp import ConnectComp
from Script.module.ReadImage import ReadImage
from Script.module.TransfromImage import TransfromImage
from Script.module.Inference import Inference
from Script.module.Server4Yolop import ServerPerception
from queue import Queue

global threadDataComp, connectComp
threadDataComp = ThreadDataComp(
    Queue(maxsize=3), 
    Queue(maxsize=3), 
    Queue(maxsize=3), 
    threading.Condition(),
    threading.Condition(),
    threading.Condition(),    
    '/home/tx2/AV-multi-machine/YOLOP/inference/videos/data_test.mp4',
    'jetson-trt/bb.trt',
    False,
    Queue()
)

connectComp = ConnectComp(
    '0.0.0.0',
    '5555',
    False
)


def main():
    FORMAT = "%(asctime)-15s : %(levelname)s : %(threadname)s - %(message)s"
    logging.basicConfig(filename="runtime.log", filemode='w', format=FORMAT)
    readImageTask = ReadImage(threadDataComp)
    tranformTask = TransfromImage(threadDataComp)
    inferenceTask = Inference(threadDataComp)
    serverPerception = ServerPerception(threadDataComp)

    print("[App]: Ready to start")

    readImageTask.start()
    tranformTask.start()
    inferenceTask.start()
    serverPerception.start()

    time.sleep(10)
    threadDataComp.isQuit = True
    while not threadDataComp.ImageQueue.empty():
        threadDataComp.ImageQueue.get()
    
    while not threadDataComp.TransformQueue.empty():
        threadDataComp.TransformQueue.get()

    readImageTask.join()
    tranformTask.join()
    serverPerception.join()

    timeSize = threadDataComp.totalTime.qsize()
    count = 0
    while not threadDataComp.totalTime.empty():
        count += threadDataComp.totalTime.get()

    print("[App]: ", threadDataComp.ImageQueue.qsize(), threadDataComp.TransformQueue.qsize(), threadDataComp.OutputQueue.qsize(), count / timeSize)

if __name__ == "__main__":
    main()

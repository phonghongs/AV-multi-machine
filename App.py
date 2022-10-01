import logging
import time
import numpy as np
import threading
import asyncio
from Script.Component.ThreadDataComp import ThreadDataComp
from Script.Component.ConnectComp import ConnectComp
from Script.module.ReadImage import ReadImage
from Script.module.TransfromImage import TransfromImage
from Script.module.Inference import Inference
from Script.module.Quanta import Quanta
from Script.module.Server4Yolop import ServerPerception
from queue import Queue

global threadDataComp, connectComp
threadDataComp = ThreadDataComp(
    Queue(maxsize=3), 
    Queue(maxsize=3), 
    asyncio.Queue(maxsize=3), 
    threading.Condition(),
    threading.Condition(),
    threading.Lock(),    
    '/home/tx2/AV-multi-machine/inference/videos/data_test.mp4',
    'jetson-trt/bb.trt',
    False,
    Queue(),
    [],
    Queue(maxsize=3),
    threading.Condition()
)

connectComp = ConnectComp(
    '0.0.0.0',
    5555,
    False
)


def main():
    FORMAT = "%(asctime)-15s : %(levelname)s : %(threadname)s - %(message)s"
    logging.basicConfig(filename="runtime.log", filemode='w', format=FORMAT)
    readImageTask = ReadImage(threadDataComp)
    tranformTask = TransfromImage(threadDataComp)
    quantaTask = Quanta(threadDataComp)
    inferenceTask = Inference(threadDataComp)
    serverPerception = ServerPerception(threadDataComp, connectComp)

    print("[App]: Ready to start")

    readImageTask.start()
    tranformTask.start()
    quantaTask.start()
    serverPerception.start()
    inferenceTask.run()

    time.sleep(10)
    threadDataComp.isQuit = True
    while not threadDataComp.ImageQueue.empty():
        threadDataComp.ImageQueue.get()
    
    while not threadDataComp.TransformQueue.empty():
        threadDataComp.TransformQueue.get()

    while not threadDataComp.QuantaQueue.empty():
        threadDataComp.QuantaQueue.get()

    threadDataComp.ImageQueue.put(None)
    threadDataComp.TransformQueue.put(None)
    threadDataComp.QuantaQueue.put(None)

    readImageTask.join()
    tranformTask.join()
    quantaTask.join()
    serverPerception.join()

    timeSize = threadDataComp.totalTime.qsize()
    count = 0
    while not threadDataComp.totalTime.empty():
        count += threadDataComp.totalTime.get()

    # np.save('testtensor.npy', threadDataComp.output[2])

    print("[App]: ", threadDataComp.ImageQueue.qsize(), threadDataComp.TransformQueue.qsize(), threadDataComp.OutputQueue.qsize(), count / timeSize)

if __name__ == "__main__":
    main()

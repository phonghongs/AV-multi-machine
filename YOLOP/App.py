import logging
import time
from Script.Component.ThreadDataComp import ThreadDataComp
from Script.module.ReadImage import ReadImage
from Script.module.TransfromImage import TransfromImage
from Script.module.Inference import Inference
from queue import Queue

global threadDataComp
threadDataComp = ThreadDataComp(
    Queue(), 
    Queue(), 
    Queue(), 
    '/home/tx2/AV-multi-machine/YOLOP/inference/videos/1.mp4',
    'jetson-trt/bb.trt',
    False
)

def main():
    # FORMAT = "%(asctime)-15s : %(levelname)s : %(threadname)s - %(message)s"
    # logging.basicConfig(filename="runtime.log", filemode='w', format=FORMAT)
    readImageTask = ReadImage(threadDataComp)
    tranformTask = TransfromImage(threadDataComp)
    inferenceTask = Inference(threadDataComp)

    readImageTask.start()
    tranformTask.start()
    inferenceTask.run()
    time.sleep(10)

    threadDataComp.isQuit = True
    readImageTask.join()
    tranformTask.join()

    print("[App]: ", threadDataComp.ImageQueue.qsize(), threadDataComp.TransformQueue.qsize(), threadDataComp.OutputQueue.qsize())

if __name__ == "__main__":
    main()
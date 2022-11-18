
from __future__ import division
from distutils.log import debug
from pickle import TRUE
from pickletools import uint8
import cv2
import numpy as np
import socket
import struct
import time
from multiple import *
import common

import threading

from Script.MqttController.MQTTController import MQTTClientController
from Script.Component.MQTTComp import MQTTComp
from Script.Component.ThreadDataComp import ThreadDataComp
from Script.Component.ConnectComp import ConnectComp
from Script.module.ClientSegment import ClientSegment
from Script.module.InferenceSeg import InferenceSegment
from Script.module.PostProcessSeg import PostProcessSeg

from queue import Queue

global threadDataComp, connectComp, mqttComp
threadDataComp = ThreadDataComp(
    Queue(maxsize=3),   #Image Queue
    Queue(maxsize=3),   #Transform Queue
    Queue(),   #Quanta Queue
    Queue(),            #Total Time Queue
    threading.Condition(),  
    threading.Condition(),
    threading.Condition(),
    threading.Lock(),    
    'inference/videos/data_test.mp4',
    'trt8_tx2/seg_16.trt',
    False,
    [],
)

mqttComp = MQTTComp(
    '192.168.1.51',
    '1883',
    'Multiple_Machine/#',
    False,
    False
)

connectComp = ConnectComp(
    '192.168.1.91',
    5555,
    False
)

def main():

    mqttController = MQTTClientController(mqttComp, threadDataComp, 'Seg')
    mqttController.client.loop_start()

    clientSegment = ClientSegment(threadDataComp, mqttComp, connectComp)
    inferenceSeg = InferenceSegment(threadDataComp)
    posprocessSeg = PostProcessSeg(threadDataComp)

    clientSegment.start()
    posprocessSeg.start()
    inferenceSeg.run()

    # inferenceSeg.join()
    inferenceSeg.delInstance()
    posprocessSeg.join()
    clientSegment.join()

    mqttController.client.loop_stop()

if __name__ == "__main__":
    main()


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
from Script.module.PlanningSystem import PlanningSystem
from Script.Utils import PareSystemConfig
from queue import Queue

global threadDataComp, connectComp, mqttComp

def SetupConfig(config:PareSystemConfig):
    global threadDataComp, connectComp, mqttComp
    threadDataComp = ThreadDataComp(
        Queue(maxsize=3),   #Image Queue
        Queue(maxsize=3),   #Transform Queue
        Queue(maxsize=3),   #Quanta Queue
        Queue(),            #Total Time Queue
        threading.Condition(),  
        threading.Condition(),
        threading.Condition(),
        threading.Lock(),    
        config.clientSegmentCfg.videoSource,
        config.clientSegmentCfg.modelPath,
        False,
        [],
    )

    mqttComp = MQTTComp(
        config.mqttCfg.brokerIP,
        config.mqttCfg.brokerPort,
        config.mqttCfg.mqttTopic,
        config.mqttCfg.controlTopic,
        config.mqttCfg.timestampTopic,
        config.mqttCfg.timestampProcessTopic,
        0,
        0,
        False,
        False,
        config.mqttCfg.isTimeStamp
    )

    connectComp = ConnectComp(
        config.clientSegmentCfg.serverIP,
        5555,
        False
    )

def main():
    global threadDataComp, connectComp, mqttComp
    config = PareSystemConfig('config.cfg')
    if (not config.isHaveConfig):
        print("[MasterController]: Pareconfig error")
        exit()

    SetupConfig(config)
    mqttController = MQTTClientController(mqttComp, threadDataComp, 'Seg')
    mqttController.client.loop_start()

    clientSegment = ClientSegment(threadDataComp, mqttComp, connectComp)
    inferenceSeg = InferenceSegment(threadDataComp)
    posprocessSeg = PostProcessSeg(threadDataComp)
    planningSeg = PlanningSystem(threadDataComp, mqttController)

    clientSegment.start()
    posprocessSeg.start()
    planningSeg.start()
    inferenceSeg.run()

    # inferenceSeg.join()
    inferenceSeg.delInstance()
    posprocessSeg.join()
    clientSegment.join()
    planningSeg.join()

    mqttController.client.loop_stop()

if __name__ == "__main__":
    main()

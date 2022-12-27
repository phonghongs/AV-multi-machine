
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
# import gc
import linecache
import os
# import tracemalloc

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

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print("#%s: %s:%s: %.1f KiB"
              % (index, frame.filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

def main():
    global threadDataComp, connectComp, mqttComp
    config = PareSystemConfig('config.cfg')
    if (not config.isHaveConfig):
        print("[MasterController]: Pareconfig error")
        exit()

    # tracemalloc.start()
    # gc.enable()
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

    # snapshot = tracemalloc.take_snapshot()
    # display_top(snapshot)

    mqttController.client.loop_stop()

if __name__ == "__main__":
    main()

import numpy as np
import json
import time
import threading
from queue import Queue
from Script.Component.ThreadDataComp import ThreadDataComp

class BicycleModel(threading.Thread):
    def __init__(self,  _threadDataComp: ThreadDataComp):
        threading.Thread.__init__(self, args=(), kwargs=None)
        self.inputQueue = Queue()
        self.threadDataComp = _threadDataComp
        self.daemon = True
        self.ouput = 0
        print("[BicycleModel]: Start converting data from file")
        self.steeringData = []
        for i in range(5, 16, 5):
            afterConvert = dict()
            with open(f"Script/Vehicles/planningData_{i}_kmh.txt", "r") as f:
                data = json.load(f)
                key = data.keys()
                for keyelement in key:
                    afterConvert[keyelement] = np.array(data[keyelement])
                self.steeringData.append(afterConvert)
        print("[BicycleModel]: the Converter is completed")


    def GetDataAtSpeed(self, speedInput):
        # Calib speed Input to [5, 15]
        if speedInput < 5:
            speedInput = 5
        elif speedInput > 15:
            speedInput = 15
        
        fileIndex = int(speedInput / 3) - 1

        if (fileIndex > 2):
            fileIndex = 2
        elif (fileIndex < 0):
            fileIndex = 0

        return self.steeringData[fileIndex]
    

    # speed : km/h ; xWaypoint, yWaypoint : m
    #                 x+
    #                 ^
    #                 |
    #                 |
    #                 |
    #                 |
    #                 |
    # y+ <------------------------
    #                 0

    def GetOptimizeSteering(self, speed, xWaypoint, yWaypoint):
        pre = time.time()
        steeringData = self.GetDataAtSpeed(speed)
        angleKey = steeringData.keys()

        minError = 9999
        optimizeAngle = 0
        for steerElement in angleKey:
            yResult = []
            cubic_equation = np.poly1d(steeringData[steerElement])
            for x in xWaypoint:
                yResult.append(cubic_equation(x))
            meanError = (np.square(np.array(yWaypoint) - np.array(yResult))).mean(axis=0)
            if (meanError < minError):
                minError = meanError
                optimizeAngle = steerElement
                # print(steeringData[steerElement], minError)
        
        # print(f"[BicycleModel] Optimize angle: {optimizeAngle}, error: {minError}, time: {time.time()  - pre}")
        return int(optimizeAngle)

    def run(self):
        while not self.threadDataComp.isQuit:
            dataInput = self.inputQueue.get()
            speed = dataInput[0]
            x = dataInput[1]
            y = dataInput[2]
            self.ouput = self.GetOptimizeSteering(speed, x, y)
import string
import paho.mqtt.client as mqtt #import the client1
import time
import logging
import keyboard
import configparser
import json
import numpy as np
import cv2
import threading
from Script.Vehicles.car import Car
from Script.Utils import PareSystemConfig
from threading import Lock
from queue import Queue
from Script.Vehicles.Bicycle_model import BicycleModel
from Script.Component.ThreadDataComp import ThreadDataComp

class MQTTClientController():
    def __init__(self, _clientName: string, lockMessage : Lock, _threadDataComp: ThreadDataComp):
        self.client = mqtt.Client(_clientName)
        self.client.connect('127.0.0.1')
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_connect_fail = self.on_connect_fail
        self.client.on_message = self.on_message
        self.lock = lockMessage
        self.isConnect = False
        self.publishTopic = "Multiple_Machine/Master"
        self.controlTopic = "Control"
        self.resultContour = []
        self.threadDataComp = _threadDataComp

    # The callback for when the client receives a CONNACK response from the server.
    def on_connect(self, client, userdata, flags, rc):
        # reconnect then subscriptions will be renewed.
        self.isConnect = True
        client.subscribe(self.controlTopic)
    
    def on_disconnect(self, client, userdata, rc):
        print("Dis-Connect")
    
    def on_connect_fail(self, client, userdata, rc):
        print("Faild")
    
    # The callback for when a PUBLISH message is received from the server.
    def on_message(self, client, userdata, msg):
        # print(msg.topic+" "+str(msg.payload))
        #control format: speed_angle
        if msg.topic == self.controlTopic:
            msgContent = msg.payload.decode("utf-8")
            if len(msgContent) > 0:
                try:
                    with self.lock:
                        self.resultContour = json.loads(msgContent)
                        # print(self.resultContour.__len__())
                except:
                    print("[MQTT]: Cannot load json from message")

    def start_segment(self):
        self.client.publish(self.publishTopic, "newUDP")
    def force_stop(self):
        self.client.publish(self.publishTopic, "quit")
        self.threadDataComp.isQuit = True

global threadDataComp

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

def warpPers(xP, yP, MP):
    p1 = (MP[0][0]*xP + MP[0][1]*yP + MP[0][2]) / (MP[2][0]*xP + MP[2][1]*yP + MP[2][2])
    p2 = (MP[1][0]*xP + MP[1][1]*yP + MP[1][2]) / (MP[2][0]*xP + MP[2][1]*yP + MP[2][2])
    return [p1, p2]

def CalSteeringAngle(dataContour, M):
    try:
        if dataContour == []:
            return
        if dataContour.__len__() < 100:
            return
        preTime = time.time()
        # blank_image = np.zeros((360, 640), np.uint8)
        # for center in dataContour:
        #     cv2.circle(blank_image, (int(center[0]), int(center[1])), 1, 255, 10)
        # cv2.imshow("IMG", blank_image)
        # cv2.waitKey(1)
        size = 3
        centers = []
        xBEV = []
        partPixel = int(360 / size)
        xList, yList = [], []
        for i in range(0, size):
            part = [x for x in dataContour if (x[1] > partPixel * i) and (x[1] < partPixel*i + partPixel)]
            part = np.array(part)

            Mm = cv2.moments(part)
            if Mm["m00"] <= 0:
                continue
            cX = int(Mm["m10"] / Mm["m00"])
            cY = int(Mm["m01"] / Mm["m00"])    

            yCh, xCh = warpPers(cX, cY, M)
            centers.append([yCh, xCh])
            # Đổi tọa đổ ảnh qua tọa độ của xe, ảnh là từ trên xuống => x ảnh ~ y ảnh và ngược lại
            # y ảnh / scalenumber - 10.6665 để đưa point về trục tọa độ xe
            # x ảnh / scalenumber - 12 để đưa point lên trên trục x xe và đảo dấu lại
            yCarAxis = np.negative(yCh / scaleNumber - 10.6665)  
            xCarAxis = np.negative(xCh / scaleNumber - 12)
            # print(xCarAxis, yCarAxis)
            xList.append(xCarAxis)    #640 / 30, / 2 = 10.6665 (center)
            yList.append(yCarAxis)

        model.inputQueue.put([10*3.6, xList, yList])
        print("Cal", time.time() - preTime)
    except Exception as e:
        print("Wait", e)
        time.sleep(0.1)

if __name__ == "__main__":
    config = PareSystemConfig('config.cfg')

    if (not config.isHaveConfig):
        print("[MasterController]: Pareconfig error")
        exit()
    
    SetupConfig(config)
    
    src = np.float32([[0, 360], [640, 360], [0, 0], [640, 0]])
    dst = np.float32([[220, 360], [400, 360], [0, 0], [640, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    scaleNumber = 30 # scale với tỉ lệ (640 / 480) , => 32 / 18, cần lấy ở giữa

    lock = Lock()
    model = BicycleModel(threadDataComp)
    golfcart = Car(config.serialCfg.serialPort, config.serialCfg.seralBaudraet, True)
    mqttClient = MQTTClientController("Master", lock, threadDataComp)

    model.start()
    mqttClient.client.loop_start()

    pre = time.time()
    while time.time() - pre < 5 and not mqttClient.isConnect:
        pass

    while mqttClient.isConnect:
        dataReceive = []
        with lock:
            dataReceive = mqttClient.resultContour
        
        CalSteeringAngle(dataReceive, M)

        if (keyboard.is_pressed('n')):
            mqttClient.start_segment()
            time.sleep(1)
        if (keyboard.is_pressed('q')):
            mqttClient.force_stop()
            mqttClient.isConnect = False

        if  golfcart.auto == True:
            print("predicted_speed, angle: ", 60, model.ouput)
            golfcart.RunAuto(60, model.ouput)

    mqttClient.client.loop_stop()
    model.OnDestroy()
    model.join()

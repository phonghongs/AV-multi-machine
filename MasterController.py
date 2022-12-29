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
        self.publishTopic = "Multiple_Machine"
        self.controlTopic = "Control"
        self.timestampTopic = "Control/timestamp"
        self.timestampProcessTopic = "Control/timestamp/process"
        self.timestamp = 0
        self.timestampProcess = 0
        self.pre_time = time.time()
        self.resultContour = []
        self.threadDataComp = _threadDataComp

    # The callback for when the client receives a CONNACK response from the server.
    def on_connect(self, client, userdata, flags, rc):
        # reconnect then subscriptions will be renewed.
        self.isConnect = True
        client.subscribe(self.controlTopic)
        client.subscribe(self.timestampTopic)
        client.subscribe(self.timestampProcessTopic)
    
    def on_disconnect(self, client, userdata, rc):
        print("Dis-Connect")
    
    def on_connect_fail(self, client, userdata, rc):
        print("Faild")
    
    # The callback for when a PUBLISH message is received from the server.
    def on_message(self, client, userdata, msg):
        # print(msg.topic+" "+str(msg.payload))
        #control format: speed_angle
        msgContent = msg.payload.decode("utf-8")
        if msg.topic == self.controlTopic:
            if len(msgContent) > 0:
                try:
                    with self.lock:
                        self.resultContour = json.loads(msgContent)
                        print(self.resultContour.__len__(), time.time() - self.pre_time)
                        self.pre_time = time.time()
                except Exception as ex:
                    print("[MQTT]: Cannot load json from message", ex)

        elif msg.topic == self.timestampTopic:
            self.timestamp = float(msgContent)
        elif msg.topic == self.timestampProcessTopic:
            self.timestampProcess = float(msgContent)
            # print(timestampProcess)

    def start_segment(self):
        self.client.publish(self.publishTopic, "newUDP")
    def force_stop(self):
        self.client.publish(self.publishTopic, "quit")
        self.threadDataComp.isQuit = True

global threadDataComp

def SetupConfig(config:PareSystemConfig):
    global threadDataComp
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

        if vizualize:
            blank_image = np.zeros((height, width), np.uint8)
            for center in dataContour:
                cv2.circle(blank_image, (int(center[0]), int(center[1])), 1, 255, 10)
        # cv2.imshow("IMG", blank_image)
        # cv2.waitKey(1)
        size = 3
        centers = []
        xBEV = []
        partPixel = int(height / size)
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
            yCh = cX
            xCh = cY
            # print("A", cX, cY)
            # print("B", yCh, xCh)
            if vizualize:
                cv2.circle(blank_image, (int(yCh), int(xCh)), 1, 255, 10)
            centers.append([yCh, xCh])
            # Đổi tọa đổ ảnh qua tọa độ của xe, ảnh là từ trên xuống => x ảnh ~ y ảnh và ngược lại
            # y ảnh / scalenumber - 10.6665 để đưa point về trục tọa độ xe
            # x ảnh / scalenumber - 12 để đưa point lên trên trục x xe và đảo dấu lại
            yCarAxis = np.negative(yCh / scaleNumber - (width / scaleNumber) / 2)     # 640 / 25 / 2 = 12.8  
            xCarAxis = np.negative(xCh / scaleNumber - (height / scaleNumber)) + cameraToCar  # 270 / 25 = 10.8
            # print(xCarAxis, yCarAxis)
            xList.append(xCarAxis)
            yList.append(yCarAxis)
        print("_____________________________________")
        xList.append(0)
        yList.append(0)

        if vizualize:
            cv2.putText(blank_image, f"{model.ouput} || {mqttClient.timestamp - mqttClient.timestampProcess - time.time() + preTime}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)
            result = cv2.cvtColor(blank_image, cv2.COLOR_BGR2RGB)
            cv2.imshow("IMG", blank_image)
            # # outputs.write(result)
            cv2.waitKey(1)
        model.inputQueue.put([5*3.6, xList, yList])
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
    scaleNumber = 25 # scale với tỉ lệ (640 / 360) , => 32 / 18, cần lấy ở giữa
    width, height = 640, 270 # 360 - 90 = 270 : 90 : 0 -> 270 is 11m
    cameraToCar = 1.5   # m
    vizualize = True
    outputs = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640, 360))

    lock = Lock()
    model = BicycleModel(threadDataComp)
    golfcart = Car(config.serialCfg.serialPort, config.serialCfg.seralBaudraet, 70, config.serialCfg.isTest)
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
            print("predicted_speed, angle: ", 70, model.ouput)
            golfcart.RunAuto(70, model.ouput*0.35)

    mqttClient.client.loop_stop()
    model.OnDestroy()
    model.join()
    outputs.release()

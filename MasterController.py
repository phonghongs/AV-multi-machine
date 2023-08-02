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
                        # data = json.loads(msgContent)
                        # print(data, time.time() - self.pre_time)
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
        config.mqttCfg.processTime,
        [],
    )

def warpPers(xP, yP, MP):
    p1 = (MP[0][0]*xP + MP[0][1]*yP + MP[0][2]) / (MP[2][0]*xP + MP[2][1]*yP + MP[2][2])
    p2 = (MP[1][0]*xP + MP[1][1]*yP + MP[1][2]) / (MP[2][0]*xP + MP[2][1]*yP + MP[2][2])
    return [p1, p2]

def CalSteeringAngle(dataContour):
    try:
        if dataContour == []:
            return
        # if dataContour.__len__() < 2:
        #     return
        preTime = time.time()
        
        blank_image = np.zeros((270, 640), np.uint8)
        centers = dataContour[2]
        for center in centers:
            cv2.circle(blank_image, (int(center[0]), int(center[1])), 1, 255, 10)

        for cnt in dataContour[3]:
            cv2.circle(blank_image, (int(cnt[0]), int(cnt[1])), 1, 255, 10)

        cv2.putText(blank_image, f"{model.ouput} || { - time.time() + preTime}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2, cv2.LINE_AA)
        model.inputQueue.put([15*3.6, dataContour[0], dataContour[1]])

        cv2.imshow("IMG", blank_image)
        cv2.waitKey(1)
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
        
        CalSteeringAngle(dataReceive)

        if (keyboard.is_pressed('n')):
            mqttClient.start_segment()
            time.sleep(1)
        if (keyboard.is_pressed('q')):
            mqttClient.force_stop()
            mqttClient.isConnect = False

        if  golfcart.auto == True:
            print("predicted_speed, angle: ", 70, model.ouput)
            golfcart.RunAuto(70, model.ouput*0.75)

    mqttClient.client.loop_stop()
    model.OnDestroy()
    model.join()
    outputs.release()

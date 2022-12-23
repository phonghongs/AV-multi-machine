import string
import paho.mqtt.client as mqtt #import the client1
import time
import logging
from Script.Component.MQTTComp import MQTTComp
from Script.Component.ThreadDataComp import ThreadDataComp
from enum import Enum

class PublishType(Enum):
    CONTROL = 1,
    TIMESTAMP = 2,
    TIMESTAMPPROCESS = 3

class MQTTClientController():
    def __init__(self, _mqttComp: MQTTComp, _threadDataComp: ThreadDataComp, _clientName: string):
        self.mqttComp = _mqttComp
        self.threadDataComp = _threadDataComp
        self.client = mqtt.Client(_clientName)

        self.client.connect(self.mqttComp.brokerIP)
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_connect_fail = self.on_connect_fail
        self.client.on_message = self.on_message

    # def __del__(self):
    #     self.client.loop_stop()

    def Block4CheckConnection(self, waitingTime: int):
        pre = time.time()
        while (time.time() - pre < waitingTime):
            pass
        if (not self.mqttComp.connectStatus):
            return False
        return True

    # The callback for when the client receives a CONNACK response from the server.
    def on_connect(self, client, userdata, flags, rc):
        # logging.info("Connected with result code "+str(rc))
        print("Connected with result code "+str(rc))
        self.mqttComp.connectStatus = True
        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        client.subscribe(self.mqttComp.commandTopic)
        client.subscribe(self.mqttComp.timestampTopic)

    def on_disconnect(self, client, userdata, rc):
        self.mqttComp.connectStatus = False
        self.threadDataComp.isQuit = False
    
    def on_connect_fail(self, client, userdata, rc):
        self.mqttComp.connectStatus = False
        self.threadDataComp.isQuit = False

    # The callback for when a PUBLISH message is received from the server.
    def on_message(self, client, userdata, msg):
        # print(msg.topic+" "+str(msg.payload))
        msgContent = msg.payload.decode("utf-8")
        
        if (msg.topic == self.mqttComp.timestampTopic):
            self.mqttComp.timestampValue = float(msgContent)
        if (msg.topic == self.mqttComp.commandTopic):
            if (msgContent == "newUDP"):
                print("NewUDP")
                self.mqttComp.createUDPTask = True

            elif (msgContent == "quit"):
                print("quit")
                self.threadDataComp.isQuit = True
                while not self.threadDataComp.ImageQueue.empty():
                    self.threadDataComp.ImageQueue.get()
                
                while not self.threadDataComp.TransformQueue.empty():
                    self.threadDataComp.TransformQueue.get()

                while not self.threadDataComp.QuantaQueue.empty():
                    self.threadDataComp.QuantaQueue.get()

                self.threadDataComp.ImageQueue.put(None)
                self.threadDataComp.TransformQueue.put(None)
                self.threadDataComp.QuantaQueue.put(None)

    # def publish_controller(self, drivableSpace):
    #     self.client.publish(self.mqttComp.controlTopic, drivableSpace)
    
    # def publish_TimeStamp(self, timestamp):
    #     self.client.publish(self.mqttComp.timestampTopic, timestamp)
    
    def publish_message(self, type : PublishType, message):
        topic = "NULL"

        if (type == PublishType.CONTROL):
            topic = self.mqttComp.controlTopic
        elif (type == PublishType.TIMESTAMP):
            topic = self.mqttComp.timestampTopic
        elif (type == PublishType.TIMESTAMPPROCESS):
            topic = self.mqttComp.timestampProcessTopic

        self.client.publish(topic, message)

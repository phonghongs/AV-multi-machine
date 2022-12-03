import string
import paho.mqtt.client as mqtt #import the client1
import time
import logging
import keyboard
import configparser
from Script.Vehicles.car import Car
from Script.Utils import PareSystemConfig

class MQTTClientController():
    def __init__(self, _clientName: string):
        self.client = mqtt.Client(_clientName)
        self.client.connect('127.0.0.1')
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_connect_fail = self.on_connect_fail
        self.client.on_message = self.on_message
        self.isConnect = False
        self.publishTopic = "Multiple_Machine/Master"
        self.controlTopic = "Control"
        self.resultSpeed = 1
        self.resultSteering = 1

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
        print(msg.topic+" "+str(msg.payload))
        #control format: speed_angle
        if msg.topic == self.controlTopic:
            msgContent = msg.payload.decode("utf-8").split("_")
            if len(msgContent) > 0:
                self.resultSpeed = (msgContent[0])
                self.resultSteering = (msgContent[1])

    def start_segment(self):
        self.client.publish(self.publishTopic, "newUDP")
    def force_stop(self):
        self.client.publish(self.publishTopic, "quit")


if __name__ == "__main__":
    config = PareSystemConfig('config.cfg')

    if (not config.isHaveConfig):
        print("[MasterController]: Pareconfig error")
        exit()
    
    golfcart = Car(config.serialCfg.serialPort, config.serialCfg.seralBaudraet, True)
    mqttClient = MQTTClientController("Master")
    mqttClient.client.loop_start()

    pre = time.time()
    while time.time() - pre < 5 and not mqttClient.isConnect:
        pass
     
    while mqttClient.isConnect:
        if (keyboard.is_pressed('n')):
            mqttClient.start_segment()
            time.sleep(1)
        if (keyboard.is_pressed('q')):
            mqttClient.force_stop()
            mqttClient.isConnect = False

        if  golfcart.auto == True:
            print("predicted_speed, angle: ", mqttClient.resultSpeed, mqttClient.resultSteering)
            golfcart.RunAuto(mqttClient.resultSpeed, mqttClient.resultSteering)

    mqttClient.client.loop_stop()

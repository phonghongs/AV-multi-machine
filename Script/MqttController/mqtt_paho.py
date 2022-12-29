# import paho.mqtt.client as mqtt #import the client1

# # The callback for when the client receives a CONNACK response from the server.
# def on_connect(client, userdata, flags, rc):
#     print("Connected with result code "+str(rc))

#     # Subscribing in on_connect() means that if we lose the connection and
#     # reconnect then subscriptions will be renewed.
#     client.subscribe("house/#")

# # The callback for when a PUBLISH message is received from the server.
# def on_message(client, userdata, msg):
#     print(msg.topic+" "+str(msg.payload))

# broker_address="192.168.1.51" 
# client = mqtt.Client("P2") #create new instance

# client.on_connect = on_connect
# client.on_message = on_message

# client.connect(broker_address)

# # Blocking call that processes network traffic, dispatches callbacks and
# # handles reconnecting.
# # Other loop*() functions are available that give a threaded interface and a
# # manual interface.
# client.loop_forever()


import string
import paho.mqtt.client as mqtt #import the client1
import time
import logging

class MQTTClientController():
    def __init__(self, _clientName: string):
        self.client = mqtt.Client(_clientName)

        self.client.connect('192.168.1.51')
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_connect_fail = self.on_connect_fail
        self.client.on_message = self.on_message

    # def Block4CheckConnection(self, waitingTime: int):
    #     pre = time.time()
    #     while (time.time() - pre < waitingTime):
    #         pass
    #     if (not self.mqttComp.connectStatus):
    #         return False
    #     return True

    # The callback for when the client receives a CONNACK response from the server.
    def on_connect(self, client, userdata, flags, rc):
        logging.info("Connected with result code "+str(rc))
        # print("Connected with result code "+str(rc))
        # reconnect then subscriptions will be renewed.
        client.subscribe("Multiple_Machine/#")

    def on_disconnect(self, client, userdata, rc):
        print("Dis-Connect")
    def on_connect_fail(self, client, userdata, rc):
        print("Faild")

    # The callback for when a PUBLISH message is received from the server.
    def on_message(self, client, userdata, msg):
        print(msg.topic+" "+str(msg.payload))


mqttClient = MQTTClientController("ABC")

mqttClient.client.loop_start()
time.sleep(10)
mqttClient.client.loop_stop()

from operator import truediv
import threading
import time
import cv2
import logging
import numpy as np
import asyncio
import socket
import struct
from Script.Component.ThreadDataComp import ThreadDataComp
from Script.Component.ConnectComp import ConnectComp
from Script.Component.MQTTComp import MQTTComp
from random import randrange

class FrameSegment(object):
    def __init__(self, loop, client):
        self.loop = loop
        self.client = client

    async def udp_frame(self, img, timestamp):
        """
        Compress image and Break down
        into data segments
        """
        # print(type(timestamp))
        length = struct.pack('>Q', len(img))
        timestampTransmit = struct.pack('>d', timestamp)

        # sendall to make sure it blocks if there's back-pressure on the socket
        await self.loop.sock_sendall(self.client, length)
        await self.loop.sock_sendall(self.client, timestampTransmit)
        await self.loop.sock_sendall(self.client, img)

class ServerPerception(threading.Thread):
    def __init__(self, _threadDataComp: ThreadDataComp, _connectComp: ConnectComp, _mqttComp: MQTTComp):
        threading.Thread.__init__(self, args=(), kwargs=None)
        self.threadDataComp = _threadDataComp
        self.connectComp = _connectComp
        self.mqttComp = _mqttComp
        self.daemon = True
        self.loop = asyncio.get_event_loop()

    def delInstance(self):
        print('received stop signal, cancelling tasks...')
        for task in asyncio.Task.all_tasks():
            task.cancel()
        print('bye, exiting in a minute...')    

    def run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.run_server())

    async def run_server(self):
        print(threading.currentThread().getName())
    
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((self.connectComp.serverIP, self.connectComp.serverPort))
        server.listen(8)
        server.setblocking(False)
        loop = asyncio.get_event_loop()

        while not self.threadDataComp.isQuit:
            # if (self.mqttComp.createUDPTask):
            try:
                print("[Server4Yolop]: Create socket")
                client, _ = await loop.sock_accept(server)
                loop.create_task(self.handle_client(client))
                self.mqttComp.createUDPTask = False
            except Exception as ex:
                print("[Server] ", ex)


    async def handle_client(self, client):
        loop = asyncio.get_event_loop()
        request = None
        fs = FrameSegment(loop, client)
        print("[Server]: In")
        while request != 'quit' or not self.threadDataComp.isQuit:
            request = (await loop.sock_recv(client, 1024)).decode('utf8')

            with self.threadDataComp.OutputCondition:
                output = self.threadDataComp.output
            
            # print(len(self.threadDataComp.output))
            [outcache, timestamp] = output
            output_udp = outcache

            # print("OK", len(output), type(output[2]), output[2].dtype)

            if (request == 'JETSON1'):
                await fs.udp_frame(output_udp[2].tobytes(), timestamp)
            elif (request == 'JETSON2'):
                stackOutput = output_udp.flatten().tobytes()
                arraySize = f"{len(output_udp[0])}|||"
                data = bytes(arraySize, 'ascii') + stackOutput
                await fs.udp_frame(data, timestamp)
            elif request != 'quit':
                await fs.udp_frame(output_udp[0], timestamp)

        client.close()

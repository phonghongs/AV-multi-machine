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


class FrameSegment(object):
    def __init__(self, loop, client):
        self.loop = loop
        self.client = client

    async def udp_frame(self, img):
        """
        Compress image and Break down
        into data segments
        """
        length = struct.pack('>Q', len(img))

        # sendall to make sure it blocks if there's back-pressure on the socket
        await self.loop.sock_sendall(self.client, length)
        await self.loop.sock_sendall(self.client, img)


class ServerPerception(threading.Thread):
    def __init__(self, _threadDataComp: ThreadDataComp, _connectComp: ConnectComp):
        threading.Thread.__init__(self, args=(), kwargs=None)
        self.threadDataComp = _threadDataComp
        self.connectComp = _connectComp
        self.daemon = True
        self.loop = asyncio.get_event_loop()

    def run(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.run_server())

    async def run_server(self):
        print(threading.currentThread().getName())

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((self.connectComp.serverIP, self.connectComp.serverPort))
        server.listen(8)
        server.setblocking(False)
        server.settimeout(3)

        while not self.threadDataComp.isQuit:
            client, _ = await self.loop.sock_accept(server)
            print(client)
            # self.loop.create_task(self.handle_client(client))

    async def handle_client(self, client):
        request = None
        fs = FrameSegment(self.loop, client)

        while request != 'quit' or not self.threadDataComp.isQuit:
            request = (await self.loop.sock_recv(client, 1024)).decode('utf8')

            with self.threadDataComp.OutputCondition:
                self.threadDataComp.OutputCondition.wait()
            output = self.threadDataComp.OutputQueue.get(timeout=1)

            if output is None:
                print("[TransFromImage] Error when get Image in queue")
                self.threadDataComp.isQuit = True
                break

            if (request == 'JETSON1'):
                await fs.udp_frame(output[2])
            elif (request == 'JETSON2'):
                stackOutput = output.flatten()
                arraySize = f"{len(output[0])}|||"
                data = bytes(arraySize, 'ascii') + stackOutput
                await fs.udp_frame(data)
            elif request != 'quit':
                await fs.udp_frame(output[0])

        client.close()


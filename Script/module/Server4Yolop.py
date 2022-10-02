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
        loop = asyncio.get_event_loop()

        while not self.threadDataComp.isQuit:
            try:
                client, _ = await loop.sock_accept(server)
                loop.create_task(self.handle_client(client))
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
                outcache = self.threadDataComp.output
            output = outcache

            # print("OK", len(output), type(output[2]), output[2].dtype)

            if (request == 'JETSON1'):
                await fs.udp_frame(output[2].tobytes())
            elif (request == 'JETSON2'):
                stackOutput = output.flatten().tobytes()
                arraySize = f"{len(output[0])}|||"
                data = bytes(arraySize, 'ascii') + stackOutput
                await fs.udp_frame(data)
            elif request != 'quit':
                await fs.udp_frame(output[0])

        client.close()

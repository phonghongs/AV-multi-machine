import asyncio, socket
from curses import raw
import re
import datetime
from threading import Lock
import cv2
import threading
import struct
import math
import numpy as np
import time

global raw_image, done
blank_image = np.zeros((480,640,3), np.uint8)
raw_image = []
done = False

ip = "0.0.0.0"
port = 5555

raw_image = np.load('testtensor.npy')
a = np.load('tensor8b_0.npy')
b = np.load('tensor8b_1.npy')
c = np.hstack((a, b))
print(c.shape)
c = c.tobytes()

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
        await loop.sock_sendall(self.client, length)
        await loop.sock_sendall(self.client, img)


async def handle_client(client):
    global raw_image

    loop = asyncio.get_event_loop()
    request = None

    fs = FrameSegment(loop, client)

    while request != 'quit':
        request = (await loop.sock_recv(client, 1024)).decode('utf8')

        if (request == 'JETSON1'):
            await fs.udp_frame(raw_image.tobytes())
        elif (request == 'JETSON2'):
            arraySize = f"{len(a)}|||"
            data = bytes(arraySize, 'ascii') + c
            await fs.udp_frame(data)
        elif request != 'quit':
            await fs.udp_frame(raw_image)
        
    client.close()


async def run_server():
    global done
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((ip, port))
    server.listen(8)
    server.setblocking(False)
    loop = asyncio.get_event_loop()

    while not done:
        client, _ = await loop.sock_accept(server)
        loop.create_task(handle_client(client))

loop = asyncio.get_event_loop()

def f(loop):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_server())

t = threading.Thread(target=f, args=(loop,))
t.start()  

print("OK")
import threading
import time
import cv2
import logging
import socket
import struct
import numpy as np
from lib.utils.augmentations import letterbox_for_img
from Script.Component.ThreadDataComp import ThreadDataComp
from Script.Component.MQTTComp import MQTTComp
from Script.Component.ConnectComp import ConnectComp

class ClientSegment(threading.Thread):
    def __init__(self, _threadDataComp: ThreadDataComp, _mqttComp: MQTTComp, _connectComp: ConnectComp):
        threading.Thread.__init__(self, args=(), kwargs=None)
        self.threadDataComp = _threadDataComp
        self.mqttComp = _mqttComp
        self.connectComp = _connectComp
        self.daemon = True
        self.CLIENT_ID = "JETSON1"
        self.MAX_DGRAM = 2**16
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def run(self):
        print(threading.currentThread().getName())
        while not self.threadDataComp.isQuit:
            if (self.mqttComp.createUDPTask):
                self.mqttComp.createUDPTask = False
                time.sleep(0.5)
                # Set up socket
                try:
                    self.s.connect((self.connectComp.serverIP, self.connectComp.serverPort))
                except Exception as e:
                    print("[ClientSegment]: Error when connect to server : MSG :", e)
                    self.threadDataComp.isQuit = True
                    break
                
                print("[ClientSegment]: Start Client")

                timecount = 0.00001
                totalTime = 0

                while not self.threadDataComp.isQuit:
                    pre = time.time()
                    self.s.send(self.CLIENT_ID.encode('utf8'))
                    bs = self.s.recv(8)
                    (length,) = struct.unpack('>Q', bs)
                    ts = self.s.recv(8)
                    (timestamp,) = struct.unpack('>d', ts)

                    data = b''
                    while len(data) < length:
                        # doing it in batches is generally better than trying
                        # to do it all in one go, so I believe.
                        to_read = length - len(data)
                        data += self.s.recv(
                            self.MAX_DGRAM if to_read > self.MAX_DGRAM else to_read)

                    result = np.frombuffer(data, dtype=np.uint8).reshape(1, 256, 48, 80)
                    out = 0.039736519607843135*(result - 9.0).astype('float32') #dequanta

                    self.threadDataComp.ImageQueue.put([out, timestamp])
                    # print("[ClientSegment]: ", time.time() - pre)

                    timecount += 1
                    totalTime += time.time() - pre
                self.s.send("quit".encode('utf8'))
                self.s.close()
                print("[ClientSegment]: Total Time : ", totalTime/timecount)

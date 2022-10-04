
from __future__ import division
from distutils.log import debug
from pickle import TRUE
from pickletools import uint8
import cv2
import numpy as np
import socket
import struct
import time
from multiple import *

global pre, raw_image
pre = time.time()

raw_image = np.load('testtensor.npy').tobytes()

print(type(raw_image))
MAX_DGRAM = 2**16

CLIENT_ID = "JETSON1"

def main():
    # Set up socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('192.168.1.91', 5555))
    done = False

    output = cv2.VideoWriter('filename.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (1280, 720))

    pre = time.time()

    print("OK")
    while time.time() - pre < 20:
        pre = time.time()
        s.send(CLIENT_ID.encode('utf8'))
        bs = s.recv(8)
        (length,) = struct.unpack('>Q', bs)
        
        data = b''
        while len(data) < length:
            # doing it in batches is generally better than trying
            # to do it all in one go, so I believe.
            to_read = length - len(data)
            data += s.recv(
                MAX_DGRAM if to_read > MAX_DGRAM else to_read)

        result = np.frombuffer(data, dtype=np.uint8).reshape(1, 256, 48, 80)
        try:
            color_area = post_process_seg(torch.tensor(result))
            output.write(color_area)
        except Exception as e:
            print(e)

        print((time.time() - pre), data == raw_image)

    output.release()
    s.send("quit".encode('utf8'))
    s.close()

if __name__ == "__main__":
    main()

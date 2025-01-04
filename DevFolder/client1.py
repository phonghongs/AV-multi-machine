
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
import common

import threading

from Script.MqttController.MQTTController import MQTTClientController
from Script.Component.MQTTComp import MQTTComp
from Script.Component.ThreadDataComp import ThreadDataComp
from Script.Component.ConnectComp import ConnectComp
from queue import Queue

global pre, mqttComp
pre = time.time()

global threadDataComp, connectComp
threadDataComp = ThreadDataComp(
    Queue(maxsize=3),   #Image Queue
    Queue(maxsize=3),   #Transform Queue
    Queue(),   #Quanta Queue
    Queue(),            #Total Time Queue
    threading.Condition(),  
    threading.Condition(),
    threading.Condition(),
    threading.Lock(),    
    '/home/tx2/AV-multi-machine/inference/videos/VideoCamGolfCar_2_Trim_Trim.mp4',
    'jetson-trt/bb.trt',
    False,
    [],
)

mqttComp = MQTTComp(
    '192.168.1.51',
    '1883',
    'Multiple_Machine/#',
    False,
    False
)

connectComp = ConnectComp(
    '192.168.1.91',
    5555,
    False
)

MAX_DGRAM = 2**16

CLIENT_ID = "JETSON1"

def inference_seg(out2):
    '''
    input: outs[2] of backbone
    output: 1 array 
    '''
    with engine.create_execution_context() as context:
        h_inputs, h_outputs, bindings, stream = common.allocate_buffers(engine)
        h_inputs[0].host = out2
        trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=h_inputs, outputs=h_outputs, stream=stream)
        trt_outputs[0] = np.reshape(trt_outputs[0],(1,2,384,640))
    return trt_outputs[0]

def load_engine(trt_file_path, verbose=False):
    """Build a TensorRT engine from a TRT file."""
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
    print('Loading TRT file from path {}...'.format(trt_file_path))
    with open(trt_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

engine = load_engine('trt8_tx2/seg_16.trt', False)

def main():

    mqttController = MQTTClientController(mqttComp, threadDataComp, 'Seg')
    mqttController.client.loop_start()

    while not threadDataComp.isQuit:
        if (mqttComp.createUDPTask):
            mqttComp.createUDPTask = False
            time.sleep(0.5)
            # Set up socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.connect((connectComp.serverIP, connectComp.serverPort))
            done = False

            output = cv2.VideoWriter('filename.avi', 
                                cv2.VideoWriter_fourcc(*'MJPG'),
                                10, (640, 360))

            print("OK")
            preeee = time.time()
            while time.time() - preeee < 100:
                pre = time.time()
                pres = time.time()
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
                print((time.time() - pres), "buffer")
                try:
                    pres = time.time()
                    out = 0.039736519607843135*(result - 9.0).astype('float32')
                    print((time.time() - pres), "dequanta")
                    pres = time.time()
                    out_seg = inference_seg(out)
                    print((time.time() - pres), "infer")
                    pres = time.time()
                    color_area = post_process_seg(torch.tensor(out_seg))
                    print((time.time() - pres), "post")
                    output.write(color_area)
                except Exception as e:
                    print(e)

                # print((time.time() - pre))


            output.release()
            s.send("quit".encode('utf8'))
            s.close()
            break
    mqttController.client.loop_stop()

if __name__ == "__main__":
    main()

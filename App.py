import logging
import time
import numpy as np
import threading
from Script.Component.ThreadDataComp import ThreadDataComp
from Script.Component.ConnectComp import ConnectComp
from Script.Component.MQTTComp import MQTTComp
from Script.module.ReadImage import ReadImage
from Script.module.TransfromImage import TransfromImage
from Script.module.Inference import Inference
from Script.module.Quanta import Quanta
from Script.module.Server4Yolop import ServerPerception
from Script.MqttController.MQTTController import MQTTClientController
from Script.Utils import PareSystemConfig
from queue import Queue

global threadDataComp, connectComp, mqttComp

def SetupConfig(config:PareSystemConfig):
    global threadDataComp, connectComp, mqttComp
    threadDataComp = ThreadDataComp(
        Queue(maxsize=3),   #Image Queue
        Queue(maxsize=3),   #Transform Queue
        Queue(maxsize=3),   #Quanta Queue
        Queue(),            #Total Time Queue
        threading.Condition(),  
        threading.Condition(),
        threading.Condition(),
        threading.Lock(),    
        config.backboneCfg.InputSource,
        config.backboneCfg.modelPath,
        False,
        config.mqttCfg.processTime,
        [],
    )

    mqttComp = MQTTComp(
        config.mqttCfg.brokerIP,
        config.mqttCfg.brokerPort,
        config.mqttCfg.mqttTopic,
        config.mqttCfg.controlTopic,
        config.mqttCfg.timestampTopic,
        config.mqttCfg.timestampProcessTopic,
        0,
        0,
        False,
        False,
        config.mqttCfg.isTimeStamp
    )

    connectComp = ConnectComp(
        config.backboneCfg.serverIP,
        5555,
        False
    )

def main():
    global threadDataComp, connectComp, mqttComp
    FORMAT = "%(asctime)-15s : %(levelname)s : %(threadname)s - %(message)s"
    logging.basicConfig(filename="runtime.log", filemode='w', format=FORMAT)
    config = PareSystemConfig('config.cfg')
    if (not config.isHaveConfig):
        print("[MasterController]: Pareconfig error")
        exit()

    SetupConfig(config)

    mqttController = MQTTClientController(mqttComp, threadDataComp, 'TX2')
    mqttController.client.loop_start()
    readImageTask = ReadImage(threadDataComp)
    tranformTask = TransfromImage(threadDataComp, mqttController)
    quantaTask = Quanta(threadDataComp)
    inferenceTask = Inference(threadDataComp)
    serverPerception = ServerPerception(threadDataComp, connectComp, mqttComp)

    print("[App]: Ready to start")

    if (not mqttController.Block4CheckConnection(3)):
        print("Fail")
        return

    print("OK")

    readImageTask.start()
    tranformTask.start()
    quantaTask.start()
    serverPerception.start()
    inferenceTask.run()

    readImageTask.join()
    tranformTask.join()
    quantaTask.join()
    serverPerception.delInstance()
    serverPerception.join()

    mqttController.client.loop_stop()

    timeSize = threadDataComp.totalTime.qsize()
    count = 0
    # while not threadDataComp.totalTime.empty():
    #     count += threadDataComp.totalTime.get()

    # np.save('testtensor.npy', threadDataComp.output[2])

    # print("[App]: ", count / timeSize)

if __name__ == "__main__":
    main()

import time
import threading
import torchvision.transforms as transforms
from Script.Component.ThreadDataComp import ThreadDataComp
from Script.MqttController.MQTTController import MQTTClientController, PublishType

class TransfromImage(threading.Thread):
    def __init__(self,  _threadDataComp: ThreadDataComp, _mqttController: MQTTClientController):
        threading.Thread.__init__(self, args=(), kwargs=None)
        self.threadDataComp = _threadDataComp
        self.mqttController = _mqttController
        self.daemon = True

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
            transforms.Resize((384, 640))
        ])

    def run(self):
        print(threading.currentThread().getName())
        timecount = 0.00001
        totalTime = 0
        while not self.threadDataComp.isQuit:
            pre = time.time()

            # with self.threadDataComp.ImageCondition:
            #     self.threadDataComp.ImageCondition.wait()
            output = self.threadDataComp.ImageQueue.get()
            # self.mqttController.publish_TimeStamp(time.time())

            if output is None:
                print("[TransFromImage] Error when get Image in queue")
                break
            
            [getImage, timestamp] = output
            self.mqttController.publish_message(PublishType.TIMESTAMP, timestamp)

            img = self.transform(getImage[0])

            img = img.float()
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # if self.threadDataComp.TransformQueue.full():
            #     self.threadDataComp.TransformQueue.get()
            self.threadDataComp.TransformQueue.put([img, timestamp])

            # with self.threadDataComp.TransformCondition:
            #     if self.threadDataComp.TransformQueue.qsize() > 0:
            #         self.threadDataComp.TransformCondition.notifyAll()

            # self.threadDataComp.totalTime.put(time.time() - pre)

            # print("[Transform] Timer ", time.time() - pre)
            timecount += 1
            totalTime += time.time() - pre
        print("[Transform]: Total Time : ", totalTime/timecount)

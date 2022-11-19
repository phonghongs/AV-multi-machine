import time
import threading
import numpy as np
import cv2
import torchvision.transforms as transforms
from Script.Component.ThreadDataComp import ThreadDataComp
from Script.MqttController.MQTTController import MQTTClientController

def warpPers(xP, yP, MP):
    p1 = (MP[0][0]*xP + MP[0][1]*yP + MP[0][2]) / (MP[2][0]*xP + MP[2][1]*yP + MP[2][2])
    p2 = (MP[1][0]*xP + MP[1][1]*yP + MP[1][2]) / (MP[2][0]*xP + MP[2][1]*yP + MP[2][2])
    return [p1, p2]

class PlanningSystem(threading.Thread):
    def __init__(self,  _threadDataComp: ThreadDataComp, _mqttController: MQTTClientController):
        threading.Thread.__init__(self, args=(), kwargs=None)
        self.threadDataComp = _threadDataComp
        self.daemon = True
        self.mqttController = _mqttController
        self.kerel5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.src = np.float32([[0, 256], [512, 256], [0, 0], [512, 0]])
        self.dst = np.float32([[200, 256], [312, 256], [0, 0], [512, 0]])
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.output = cv2.VideoWriter('outuit2.avi', 
                                cv2.VideoWriter_fourcc(*'MJPG'),
                                10, (640, 360))
    def run(self):
        print(threading.currentThread().getName())
        timecount = 0.00001
        totalTime = 0
        while not self.threadDataComp.isQuit:
            pre = time.time()
            prepre = time.time()

            da_seg_mask = self.threadDataComp.QuantaQueue.get()

            if da_seg_mask is None:
                print("[TransFromImage] Error when get Image in queue")
                break

            color_area = np.zeros(
                (da_seg_mask.shape[0], da_seg_mask.shape[1], 1), dtype=np.uint8)
            color_area[da_seg_mask == 1] = [255]

            thre_mask = cv2.morphologyEx(color_area,cv2.MORPH_DILATE, self.kerel5)

            # conts, hier = cv2.findContours(thre_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            conts, hier = cv2.findContours(thre_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            cont = sorted(conts, key= lambda area_Index: cv2.contourArea(area_Index) , reverse=True)[0]
            # # print(cont.shape)
            # # print("=================")
            finalCont = [cnt[0] for cnt in cont.tolist()]

            # finalCont : (640, 360) : (x, y) lọc theo x từ trên xuống rồi lọc theo y từ trái qua để ra được [[6, 257], [6, 238], [13, 238], [14, 237], [14, 232], [15, 231]
            finalCont = sorted(finalCont, key= lambda y : y[1], reverse=True)
            finalCont = sorted(finalCont, key= lambda y : y[0])
            blank_image = np.zeros((da_seg_mask.shape[0], da_seg_mask.shape[1], 3), np.uint8)

            # # print(finalCont)

            upperLine = []
            upperLine.append([0, da_seg_mask.shape[0]])

            lastY = finalCont[0][1]
            for i in range(0, finalCont.__len__() - 1):
                if (finalCont[i][0] != finalCont[i+1][0]):
                    upperLine.append(finalCont[i])
            upperLine.append(finalCont[-1])
            # upperLine.append([finalCont[-1][0], heigh])
            upperLine.append([da_seg_mask.shape[1], da_seg_mask.shape[0]])
            # upperLine = sorted(upperLine, key= lambda y : y[1])
            upperLine = np.array(upperLine)
            # print(upperLine.shape, type(upperLine))
            # print(upperLine)

            cv2.fillPoly(blank_image, pts = [upperLine], color =255)
            M = cv2.moments(upperLine)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cv2.circle(blank_image, (cX, cY), 7, (255, 255, 255), -1)

            self.mqttController.publish_controller(cX, cY)
            print("[TransformImage]: ", time.time() - prepre)
            # print(output[2].dtype, type(output[2]), output[2].shape)
            with self.threadDataComp.OutputCondition:
                self.threadDataComp.output = blank_image

            # self.output.write(blank_image)
            timecount += 1
            totalTime += time.time() - pre
        print("[Quanta]: Total Time : ", totalTime/timecount)

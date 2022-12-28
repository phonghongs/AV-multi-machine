import time
import threading
import numpy as np
import cv2
import torchvision.transforms as transforms
from Script.Component.ThreadDataComp import ThreadDataComp
from Script.MqttController.MQTTController import MQTTClientController, PublishType

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
        self.src = np.float32([[0, 360], [640, 360], [0, 0], [640, 0]])
        self.dst = np.float32([[200, 360], [440, 360], [0, 0], [640, 0]])
        self.M = cv2.getPerspectiveTransform(self.src, self.dst)
        self.output = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640, 360))

    def run(self):
        print(threading.currentThread().getName())
        timecount = 0.00001
        totalTime = 0
        while not self.threadDataComp.isQuit:
            pre = time.time()

            output = self.threadDataComp.QuantaQueue.get()
            prepre = time.time()
            if output is None:
                print("[TransFromImage] Error when get Image in queue")
                break
            
            [da_seg_mask, timestamp] = output
            color_area = np.zeros(
                (da_seg_mask.shape[0], da_seg_mask.shape[1], 1), dtype=np.uint8)
            
            
            color_area[da_seg_mask == 1] = [255]
            output = cv2.cvtColor(color_area, cv2.COLOR_GRAY2RGB)
            print(color_area.shape)

            self.output.write(output)
            continue

            color_area = da_seg_mask.astype(np.uint8)
            try:
                #_____________________ Find contour of segment _____________________
                conts, hier = cv2.findContours(color_area, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                cont = sorted(conts, key= lambda area_Index: cv2.contourArea(area_Index) , reverse=True)[0]
                # # print(cont.shape)
                # # print("=================")
                finalCont = [cnt[0] for cnt in cont.tolist()]

                # finalCont : (640, 360) : (x, y) lọc theo x từ trên xuống rồi lọc theo y từ trái qua để ra được [[6, 257], [6, 238], [13, 238], [14, 237], [14, 232], [15, 231]
                finalCont = sorted(finalCont, key= lambda y : y[1], reverse=True)
                finalCont = sorted(finalCont, key= lambda y : y[0])

                #_____________________ Contour Fillter _____________________
                blank_image = np.zeros((color_area.shape[0], color_area.shape[1]), np.uint8)
                upperLine = []
                upperLine.append([0, color_area.shape[0]])
                lastY = finalCont[0][1]
                for i in range(0, finalCont.__len__() - 1):
                    if (finalCont[i][0] != finalCont[i+1][0]):
                        upperLine.append(finalCont[i])
                upperLine.append(finalCont[-1])
                upperLine.append([color_area.shape[1], color_area.shape[0]])
                upperLine = np.array(upperLine)
                #_____________________ Find segment Center  _____________________
                cv2.fillPoly(blank_image, pts = [upperLine], color =255)
                conts, hier = cv2.findContours(blank_image[90:], cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                cont = sorted(conts, key= lambda area_Index: cv2.contourArea(area_Index) , reverse=True)[0]
                finalCont = [cnt[0] for cnt in cont.tolist()]

                # output = cv2.cvtColor(blank_image, cv2.COLOR_GRAY2RGB)
                # self.output.write(output)
                # self.output.write(blank_image)
                # self.mqttController.publish_controller(str(finalCont))
                
                self.mqttController.publish_message(PublishType.CONTROL, str(finalCont))
                if (self.mqttController.IsTimeStamp()):
                    self.mqttController.publish_message(PublishType.TIMESTAMPPROCESS, str(timestamp))
                # print("[PlanningSystem]: ", time.time() - prepre, self.mqttController.mqttComp.timestampValue - timestamp)
            except Exception as e:
                print("[PlanningSystem]", e)
            # print(output[2].dtype, type(output[2]), output[2].shape)
            # with self.threadDataComp.OutputCondition:
            #     self.threadDataComp.output = blank_image

            timecount += 1
            totalTime += time.time() - pre
        print("[Quanta]: Total Time : ", totalTime/timecount)
        self.output.release()

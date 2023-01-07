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
        self.scaleNumber = 25
        self.cameraToCar = 1.5
        self.output = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640, 270))

    def run(self):
        print(threading.currentThread().getName())
        timecount = 0.00001
        totalTime = 0
        while not self.threadDataComp.isQuit:
            pre = time.time()

            output = self.threadDataComp.QuantaQueue.get()
            prepre = time.time()

            if self.threadDataComp.isTimeProcess:
                pre = time.time()

            if output is None:
                print("[TransFromImage] Error when get Image in queue")
                break

            [da_seg_mask, timestamp] = output
            # color_area = np.zeros(
            #     (da_seg_mask.shape[0], da_seg_mask.shape[1], 1), dtype=np.uint8)
            
            
            # color_area[da_seg_mask == 1] = [255]

            # for i in range(0, color_area.shape[1]):
            #     color_area[color_area.shape[0] - 1, i] = 255

            # cont, hier = cv2.findContours(color_area, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            # maxarea=0
            # indx=-1
            # for ind,cnt in enumerate(cont) :
            #     area = cv2.contourArea(cnt)
            #     if area > maxarea:
            #         indx=ind
            #         maxarea=area
            # mask_num = np.zeros(color_area.shape, np.uint8)
            # cv2.drawContours(mask_num, [cont[indx]], -1, 255, -1)

            # output = cv2.cvtColor(mask_num, cv2.COLOR_GRAY2RGB)
            # print(color_area.shape)

            # self.output.write(output)
            # continue

            color_area = da_seg_mask.astype(np.uint8)
            color_area = color_area[90:]                # 90 -> 360
            self.height = color_area.shape[0]
            self.width = color_area.shape[1]

            for i in range(0, color_area.shape[1]):
                color_area[color_area.shape[0] - 1, i] = 1
            try:
                #_____________________ Find contour of segment _____________________
                
                # finalCont = self.FillNoise(color_area)
                finalCont = self.FillNoise_v2(color_area)
                centerPoint = self.MidPointFinding(finalCont)
                centerPoint.append(finalCont)
                # blank_image = np.zeros((self.height, self.width), np.uint8)
                # for cnt in finalCont:
                #     cv2.circle(blank_image, (int(cnt[0]), int(cnt[1])), 1, 255, 10)

                # centers = centerPoint[2]
                # for center in centers:
                #     cv2.circle(blank_image, (int(center[0]), int(center[1])), 1, 255, 10)
                # output = cv2.cvtColor(blank_image, cv2.COLOR_GRAY2RGB)
                # print(output.shape)
                # self.output.write(output)

                self.mqttController.publish_message(PublishType.CONTROL, str(centerPoint))
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

    def MidPointFinding(self, input):
        size = 3
        centers = []
        xBEV = []
        partPixel = int(self.height / size)
        xList, yList = [], []
        for i in range(0, size):
            part = [x for x in input if (x[1] > partPixel * i) and (x[1] < partPixel*i + partPixel)]
            part = np.array(part)

            Mm = cv2.moments(part)
            if Mm["m00"] <= 0:
                continue
            cX = int(Mm["m10"] / Mm["m00"])
            cY = int(Mm["m01"] / Mm["m00"])    
            yCh = cX
            xCh = cY
            # print("A", cX, cY)
            # print("B", yCh, xCh)

            centers.append([yCh, xCh])
            # Đổi tọa đổ ảnh qua tọa độ của xe, ảnh là từ trên xuống => x ảnh ~ y ảnh và ngược lại
            # y ảnh / scalenumber - 10.6665 để đưa point về trục tọa độ xe
            # x ảnh / scalenumber - 12 để đưa point lên trên trục x xe và đảo dấu lại
            yCarAxis = np.negative(yCh / self.scaleNumber - (self.width / self.scaleNumber) / 2)     # 640 / 25 / 2 = 12.8  
            xCarAxis = np.negative(xCh / self.scaleNumber - (self.height / self.scaleNumber)) + self.cameraToCar * (size - i)  # 270 / 25 = 10.8
            # print(xCarAxis, yCarAxis)
            xList.append(xCarAxis)
            yList.append(yCarAxis)
            result = [xList, yList, centers]
        return result

    def FillNoise_v2(self, input):
        image = cv2.warpPerspective(input, self.M, (self.width, self.height))
        conts, hier = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cont = sorted(conts, key= lambda area_Index: cv2.contourArea(area_Index) , reverse=True)[0]
        finalCont = [cnt[0] for cnt in cont.tolist()]
        return finalCont

    def FillNoise(self, input):
        conts, hier = cv2.findContours(input, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cont = sorted(conts, key= lambda area_Index: cv2.contourArea(area_Index) , reverse=True)[0]
        # # print(cont.shape)
        # # print("=================")
        finalCont = [cnt[0] for cnt in cont.tolist()]

        # finalCont : (640, 360) : (x, y) lọc theo x từ trên xuống rồi lọc theo y từ trái qua để ra được [[6, 257], [6, 238], [13, 238], [14, 237], [14, 232], [15, 231]
        finalCont = sorted(finalCont, key= lambda y : y[1], reverse=True)
        finalCont = sorted(finalCont, key= lambda y : y[0])

        #_____________________ Contour Fillter _____________________
        blank_image = np.zeros((input.shape[0], input.shape[1]), np.uint8)
        upperLine = []
        upperLine.append([0, input.shape[0]])
        lastY = finalCont[0][1]
        for i in range(0, finalCont.__len__() - 1):
            if (finalCont[i][0] != finalCont[i+1][0]):
                upperLine.append(finalCont[i])
        upperLine.append(finalCont[-1])
        upperLine.append([input.shape[1], input.shape[0]])
        upperLine = np.array(upperLine)
        #_____________________ Find segment Center  _____________________
        cv2.fillPoly(blank_image, pts = [upperLine], color =255)
        conts, hier = cv2.findContours(blank_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        cont = sorted(conts, key= lambda area_Index: cv2.contourArea(area_Index) , reverse=True)[0]
        finalCont = [cnt[0] for cnt in cont.tolist()]

        return finalCont
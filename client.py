# Import socket module
import socket
import cv2
from matplotlib import type1font
import numpy as np
import json
import math
from numpy.polynomial import Polynomial as P
from Script.Vehicles.Bicycle_model import BicycleModel
import time

heigh = 360
width = 640
dt = 0.2  # [s]
L = 2.9  # [m]
Lr = 1.4  # [m]

scaleNumber = 30 # scale với tỉ lệ (640 / 480) , => 32 / 18, cần lấy ở giữa
model = BicycleModel()

def warpPers(xP, yP, MP):
    p1 = (MP[0][0]*xP + MP[0][1]*yP + MP[0][2]) / (MP[2][0]*xP + MP[2][1]*yP + MP[2][2])
    p2 = (MP[1][0]*xP + MP[1][1]*yP + MP[1][2]) / (MP[2][0]*xP + MP[2][1]*yP + MP[2][2])
    return [p1, p2]

class State:
    def __init__(self, x=Lr, y=0.0, yaw=0.0, v=0.0, beta=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.beta = beta


def update(state, a, delta):

    state.beta = math.atan2(Lr / L * math.tan(delta), 1.0)

    state.x = state.x + state.v * math.cos(state.yaw + state.beta) * dt
    state.y = state.y + state.v * math.sin(state.yaw + state.beta) * dt
    state.yaw = state.yaw + state.v / Lr * math.sin(state.beta) * dt
    state.v = state.v + a * dt

    #  print(state.x, state.y, state.yaw, state.v)

    return state

def findPathWithVelocity(speed):
    T = 50
    a = [0] * T
    delta = [math.radians(10.0)] * T
    #  print(a, delta)

    x = []
    y = []
    yaw = []
    v = []
    beta = []
    times = []
    times = []
    t = 0.0
    x_total = []
    y_total = []

    for i in range (-25, 26, 5):
        state = State(v=speed)

        x = []
        y = []
        yaw = []
        v = []
        beta = []
        times = []
        times = []
        t = 0.0
        
        delta = [math.radians(i)] * T

        for (ai, di) in zip(a, delta):
            t = t + dt
            state = update(state, ai, di)
            x.append(state.x)
            y.append(state.y)
            yaw.append(state.yaw)
            v.append(state.v)
            beta.append(state.beta)
            times.append(t)
        x_total.append(x)
        y_total.append(y)
    return x_total, y_total

def planning(image, M, speed):
    preTime = time.time()
    image = cv2.resize(image[170:], (width, heigh))
    # image = cv2.warpPerspective(image, M, (width, heigh))
    #_____________________ Find contour of segment _____________________
    conts, hier = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cont = sorted(conts, key= lambda area_Index: cv2.contourArea(area_Index) , reverse=True)[0]
    # # print(cont.shape)
    # # print("=================")
    finalCont = [cnt[0] for cnt in cont.tolist()]

    # finalCont : (640, 360) : (x, y) lọc theo x từ trên xuống rồi lọc theo y từ trái qua để ra được [[6, 257], [6, 238], [13, 238], [14, 237], [14, 232], [15, 231]
    finalCont = sorted(finalCont, key= lambda y : y[1], reverse=True)
    finalCont = sorted(finalCont, key= lambda y : y[0])

    #_____________________ Contour Fillter _____________________
    blank_image = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    upperLine = []
    upperLine.append([0, image.shape[0]])
    lastY = finalCont[0][1]
    for i in range(0, finalCont.__len__() - 1):
        if (finalCont[i][0] != finalCont[i+1][0]):
            upperLine.append(finalCont[i])
    upperLine.append(finalCont[-1])
    upperLine.append([image.shape[1], image.shape[0]])
    upperLine = np.array(upperLine)
    # print(upperLine.tolist())
    #_____________________ Find segment Center  _____________________
    cv2.fillPoly(blank_image, pts = [upperLine], color =255)
    conts, hier = cv2.findContours(blank_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cont = sorted(conts, key= lambda area_Index: cv2.contourArea(area_Index) , reverse=True)[0]
    finalCont = [cnt[0] for cnt in cont.tolist()]
    result = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    print(finalCont.__len__())
    size = 3
    centers = []
    xBEV = []
    partPixel = int(blank_image.shape[0] / size)
    xList, yList = [], []
    for i in range(0, size):
        part = [x for x in finalCont if (x[1] > partPixel * i) and (x[1] < partPixel*i + partPixel)]
        part = np.array(part)

        Mm = cv2.moments(part)
        if Mm["m00"] <= 0:
            continue
        cX = int(Mm["m10"] / Mm["m00"])
        cY = int(Mm["m01"] / Mm["m00"])    

        yCh, xCh = warpPers(cX, cY, M)
        centers.append([yCh, xCh])
        # Đổi tọa đổ ảnh qua tọa độ của xe, ảnh là từ trên xuống => x ảnh ~ y ảnh và ngược lại
        # y ảnh / scalenumber - 10.6665 để đưa point về trục tọa độ xe
        # x ảnh / scalenumber - 12 để đưa point lên trên trục x xe và đảo dấu lại
        yCarAxis = np.negative(yCh / scaleNumber - 10.6665)  
        xCarAxis = np.negative(xCh / scaleNumber - 12)
        # print(xCarAxis, yCarAxis)
        xList.append(xCarAxis)    #640 / 30, / 2 = 10.6665 (center)
        yList.append(yCarAxis)

    angle = - model.GetOptimizeSteering(speed*3.6, xList, yList)
    AVControl(25/3.6, angle)

    # print(speed, angle, 1 /(time.time() - preTime + 0.00001))


    img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for center in centers:
        cv2.circle(img, (int(center[0]), int(center[1])), 1, (0, 255, 0), 10)
    # x, y = findPathWithVelocity(float(speed))
    # for (a, b) in zip(x, y):
    #     for (n, m) in zip(a, b):
    #         cv2.circle(img, (int(n), int(m)), 1, (255, 0, 0), 5)


    # cv2.line(result, (int(image.shape[1]/2), image.shape[0]), (cX, cY), 255, 5)
    cv2.imshow("After", img)

MODE_1 = 18520
MODE_2 = 331

# Create a socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Define the port on which you want to connect
PORT = 11000
# connect to the server on local computer
s.connect(('127.0.0.1', PORT))

global speedcmd, anglecmd
speedcmd = 0
anglecmd = 0

def jsonObject(cmd = MODE_1):
    cmt = {}
    if cmd == MODE_1:
        cmt['Cmd'] = cmd
        cmt['Speed'] = speedcmd
        cmt['Angle'] = anglecmd
    else:
        cmt['Cmd'] = cmd
    return bytes(str(cmt), "utf-8")

def AVControl(speed, angle):
    global speedcmd, anglecmd
    speedcmd = speed
    anglecmd = angle

if __name__ == "__main__":
    src = np.float32([[0, 360], [640, 360], [0, 0], [640, 0]])
    dst = np.float32([[220, 360], [400, 360], [0, 0], [640, 0]])
    M = cv2.getPerspectiveTransform(src, dst)
    try:
        while True:
            # CMD 1
            s.sendall(jsonObject(MODE_1))
            data = s.recv(255)
            y = json.loads(data)
            # print(float(y['Speed']))
    
            #CMD 2
            s.sendall(jsonObject(MODE_2))
            data = s.recv(100000)
            try:
                image = cv2.imdecode(
                    np.frombuffer(
                        data,
                        np.uint8
                        ), -1
                    )
                color_area = cv2.inRange(image, (178, 255, 0), (179, 255, 0))
                planning(color_area, M, float(y['Speed']) / 3.6)
                # cv2.imshow("SEG", color_area)
            except Exception as er:
                print(er)
                pass
            
            #maxspeed = 90, max steering angle = 25
            # AVControl(speed=10, angle=10)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    finally:
        print('closing socket')
        s.close()



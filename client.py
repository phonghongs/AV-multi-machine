# Import socket module
import socket
import cv2
from matplotlib import type1font
import numpy as np
import json

def planning(image):
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

    #_____________________ Find segment Center  _____________________
    cv2.fillPoly(blank_image, pts = [upperLine], color =255)
    conts, hier = cv2.findContours(blank_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cont = sorted(conts, key= lambda area_Index: cv2.contourArea(area_Index) , reverse=True)[0]
    finalCont = [cnt[0] for cnt in cont.tolist()]
    result = np.zeros((image.shape[0], image.shape[1]), np.uint8)

    # for cnt in finalCont:
    #     cv2.circle(result, (cnt[0], cnt[1]), 1, 255, 10)

    centers = []
    partPixel = int(blank_image.shape[0] / 3)
    for i in range(0, 3):
        part = [x for x in finalCont if (x[1] > partPixel * i) and (x[1] < partPixel*i + partPixel)]
        part = np.array(part)

        M = cv2.moments(part)
        if M["m00"] <= 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])    

        centers.append([cX, cY])

    #_____________________ Visualize _____________________
    for center in centers:
        cv2.circle(result, (center[0], center[1]), 1, 255, 5)

    # cv2.line(result, (int(image.shape[1]/2), image.shape[0]), (cX, cY), 255, 5)
    cv2.imshow("After", image)

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
    try:
        while True:
            # CMD 1
            s.sendall(jsonObject(MODE_1))
            data = s.recv(255)
            y = json.loads(data)
            print(y)

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
                planning(color_area)
                # cv2.imshow("SEG", color_area)
            except Exception as er:
                print(er)
                pass
            
            #maxspeed = 90, max steering angle = 25
            AVControl(speed=10, angle=10)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

    finally:
        print('closing socket')
        s.close()



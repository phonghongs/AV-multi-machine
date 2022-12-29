import cv2
import time

cap = cv2.VideoCapture('/dev/video0')


# pre = time.time()
# total = 0
# i = 0
# while time.time() - pre < 5:
#     i += 1
#     time_pre = time.time()
#     _, img = cap.read()
#     total += time.time() - time_pre

while True:
    _, img = cap.read()
    cv2.imshow("IMG", img)
    if (cv2.waitKey(1) == 27):
        break


# print(img.shape, total/i)

cap.release()

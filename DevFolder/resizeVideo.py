import cv2
import numpy as np

output = cv2.VideoWriter('filename.avi', 
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, (640, 480))

cap = cv2.VideoCapture('/home/tx2/AV-multi-machine/inference/videos/VideoCamGolfCar_2_Trim_Trim.mp4')
while cap.isOpened():
    ret, img = cap.read()
    if (ret):
        img = cv2.resize(img, (640, 480))
        output.write(img)
    else:
        break
    
output.release()
cap.release()
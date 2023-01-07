import cv2
import time

cap = cv2.VideoCapture('/dev/video1')

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('record/output.avi', fourcc, 20.0, (640, 360))
video_count = 0
pre = time.time()
total = 0
i = 0
while time.time() - pre < 5:
    i += 1
    time_pre = time.time()
    _, img = cap.read()
    total += time.time() - time_pre

# pre = time.time()
# while True:
#     _, img = cap.read()
#     cv2.imshow("IMG", img)
#     if (time.time() - pre > 10):
#         out.release()
#         out = cv2.VideoWriter(f'record/output_{video_count}.avi', fourcc, 20.0, (640, 360))
#     if (cv2.waitKey(1) == 27):
#         break


# print(img.shape, total/i)

cap.release()

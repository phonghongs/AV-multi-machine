import time
import threading
import torchvision.transforms as transforms
from Script.Component.ThreadDataComp import ThreadDataComp

class TransfromImage(threading.Thread):
    def __init__(self,  _threadDataComp: ThreadDataComp):
        threading.Thread.__init__(self, args=(), kwargs=None)
        self.threadDataComp = _threadDataComp
        self.daemon = True

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
        ])

    def run(self):
        print(threading.currentThread().getName())
        while not self.threadDataComp.isQuit:
            pre = time.time()
            getImage = self.threadDataComp.ImageQueue.get()

            if getImage is None:
                print("[TransFromImage] Error when get Image in queue")
                break

            img = self.transform(getImage[0])
            img = img.float()
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            self.threadDataComp.TransformQueue.put(
                img
            )
            self.threadDataComp.totalTime.put(time.time() - pre)

            # print("[Transform] Timer ", time.time() - pre)

import numpy as np
import time


class Controller():
    def __init__(self, serial, test):
        self.serialTest = test
        self.serial = serial
        self.mode = 0
        self.gear_box = {0: 0, 1:30, 2:40, 3:50, 4:60, 5:70, 6:80, 7:90, 8:100}
        self.direction = 0
        self.speed = 0
        self.angle = 0
        self.currentAngle = 0
        self.currentSpeed = 0


    def update(self):
        self.speed = np.clip(self.speed, 0, self.gear_box[self.mode])


    def reset_angle(self):
        self.angle = 0


    def reset(self):
        self.direction = 0
        self.mode = 0
        self.speed = 0
        self.angle = 0


    def increase_mode(self):                               ## Tăng số
        self.mode = np.clip(self.mode + 1, 0, 8)
        self.update()


    def decrease_mode(self):                               ## Giảm số
        self.mode = np.clip(self.mode - 1, 0, 8)
        self.update()


    def increase_speed(self, step):                        ## Tăng tốc
        self.speed = np.clip(self.speed + step, 0, self.gear_box[self.mode])


    def decrease_speed(self, step):                        ## Giảm tốc
        self.speed = np.clip(self.speed - step, 0, self.gear_box[self.mode])
    

    def turn_right(self, step):                            ## Tăng gốc cua phải
        self.angle = np.clip(self.angle + step, -25, 25)


    def turn_left(self, step):                             ## Tăng góc cua trái
        self.angle = np.clip(self.angle - step, -25, 25)


    def get_info(self):
        return self.mode, self.direction, self.speed, self.angle


    def go_straight(self):
        if self.direction >= 0:               ## Đang đi thẳng
            self.Car_SetSpeedAngle(self.speed, self.angle, 0)
            self.direction = 1

        else:                                 ## Đang đi lùi
            self.Car_SetSpeedAngle(0, self.angle, 1)
            self.Car_SetSpeedAngle(0, self.angle, 1)
            time.sleep(0.1)
            self.Car_SetSpeedAngle(-self.speed, self.angle, 0)
            self.direction = 1


    def go_reverse(self):
        if self.direction > 0:               ## Đang đi thẳng
            self.Car_SetSpeedAngle(0, self.angle, 1)
            self.Car_SetSpeedAngle(0, self.angle, 1)
            time.sleep(0.1)
            self.Car_SetSpeedAngle(-self.speed, self.angle, 0)
            self.direction = -1

        else:                                 ## Đang đi lùi
            self.Car_SetSpeedAngle(-self.speed, self.angle, 0)  
            self.direction = -1

    def ResetDriver(self):
        Send = bytes([0xAA, 0x07, 0x01, 0xEE])
        print("Reset")

        if not self.serialTest:
            self.serial.write(Send)

    def brake(self):
        print("BRAKE")
        self.Car_SetSpeedAngle(0, self.angle, 1)
        self.reset()


    def nop(self):
        # print("NOP")
        self.Car_SetSpeedAngle(0, self.angle, 0)


    def GetRealAngle(self, newangle):
        if self.angle < newangle:
            self.angle = np.clip(self.angle + 1.5, -25, 25)
        elif self.angle > newangle:
            self.angle = np.clip(self.angle - 1.5, -25, 25)

        newangle = int(np.clip(newangle, -24, 24))
        self.angle = newangle

        return self.angle


    def Car_ChangeMaxSpeed(self, Max_speed):
        Max_speed = np.clip(Max_speed, 0 , 100)
        Send = bytes([0xAA, 0x05, int(Max_speed), 0xEE])
        print(Send)

        if not self.serialTest:
            self.serial.write(Send)


    def Car_SetSpeedAngle(self, speed, angle, allowRun):
        angle = np.clip(angle, -24, 24)
        speed = np.clip(speed, -100, 100)

        allowRun = np.clip(allowRun, 0, 1)

        sendAngle = int((angle + 25) * 4)
        sendSpeed = int(speed + 100)
        Send = bytes([0xAA, 0x04, allowRun, sendSpeed, sendAngle, 0xEE])    

        if not self.serialTest:
            self.serial.write(Send)

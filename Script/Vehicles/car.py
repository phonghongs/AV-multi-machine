from Script.Vehicles.controller import Controller
import threading
import serial
import numpy as np
import keyboard
import time


class SingletonMeta(type):

    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwds):
        with cls._lock:
            if cls not in cls._instances:
                isinstance = super().__call__(*args, **kwds)
                cls._instances[cls] = isinstance
        return cls._instances[cls]


class Car(metaclass=SingletonMeta):
    def __init__(self, serialPort, serialBaudrate, maxSpeed = 70, serialtest = True):
        self.auto = False
        self.done = False
        self.serialTest = serialtest
        self.serial = self.connect(serialPort, serialBaudrate)
        self.carController = Controller(self.serial, self.serialTest)
        self.carController.Car_ChangeMaxSpeed(maxSpeed)

        self.main_ThreadKeyBoard = threading.Thread(target= self.Thread_read_key_board)
        self.main_ThreadKeyBoard.start()


    def connect(self, port, baudrate):
        if not self.serialTest:
            ser = serial.Serial(
                port= port,
                baudrate = baudrate,
            )
            if (ser):
                print("Serial communication success!!")
                return ser
        else:
            return 0


    def RunAuto(self, speed, angle):
        newangle = self.carController.GetRealAngle(angle)
        self.carController.Car_SetSpeedAngle(self.carController.speed , newangle, 0)
        print(self.carController.speed , newangle)


    def Thread_read_key_board(self):

        while not self.done:
            if self.auto == False:
                print("Car controller: ", self.carController.get_info())
                if keyboard.is_pressed('space') :
                    self.carController.increase_mode()

                if keyboard.is_pressed('ctrl') :
                    self.carController.decrease_mode()

                # elif keyboard.is_pressed('w'):
                #     self.carController.increase_speed(2)

                # elif keyboard.is_pressed('s') :   
                #     self.carController.decrease_speed(2)
                
                elif keyboard.is_pressed('d') :
                    self.carController.turn_right(1)

                elif keyboard.is_pressed('a') :
                    self.carController.turn_left(1)

                if keyboard.is_pressed('up'):
                    self.carController.go_straight()  

                elif keyboard.is_pressed('down'):
                    self.carController.go_reverse()  

                elif keyboard.is_pressed('enter'):
                    self.carController.brake()

                else:
                    self.carController.nop()

            if keyboard.is_pressed('right shift') and self.auto == True:
                self.auto = False
                self.carController.brake()  

            elif keyboard.is_pressed('tab'):
                print("HRERERERE")
                if self.auto == False:
                    print("Auto")
                    self.auto = True     
                else:       
                    print("Manual")   
                    self.auto = False
                    self.carController.brake()
                time.sleep(1)
            
            elif keyboard.is_pressed('r'):
                    self.carController.ResetDriver()
            elif keyboard.is_pressed('w'):
                self.carController.increase_speed(2)
                # self.carController.speed = np.clip(self.carController.speed + 2, 0, 100)
            elif keyboard.is_pressed('s'):
                self.carController.decrease_speed(2)
                # self.carController.speed = np.clip(self.carController.speed - 2, 0, 100)
            elif keyboard.is_pressed('esc'):
                self.done = True

            time.sleep(0.1)


    def EndProcess(self):
        self.done = True
        self.main_ThreadKeyBoard.join()

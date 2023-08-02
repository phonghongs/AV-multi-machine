#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Kinematic Bicycle Model

author Atsushi Sakai
"""

import math
import time
import json
dt = 0.1  # [s]
L = 2.9  # [m]
Lr = 1.4  # [m]


class State:

    def __init__(self, x=Lr, y=0.0, yaw=0.0, v=0.0, beta=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.beta = beta


def update(state, a, delta):

    state.x = state.x + state.v * math.cos(state.yaw) * dt
    state.y = state.y + state.v * math.sin(state.yaw) * dt
    state.yaw = state.yaw + state.v / (L - Lr)* delta * dt
    state.v = state.v + a * dt

    return state


if __name__ == '__main__':
    print("start Kinematic Bicycle model simulation")
    # import matplotlib.pyplot as plt
    import numpy as np

    T = 20
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
    

    result = []

    for index in range (5, 16, 5):
        x_total = []
        y_total = []
        resultInSpeed = dict()
        for i in range (-20, 21):
            state = State(v=index/ 3.6)

            x = [0]
            y = [0]
            yaw = []
            v = []
            beta = []
            times = []
            times = []
            t = 0.0
            a = [0] * T
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
            pfit = np.polyfit(x, y, 3)
        
            resultInSpeed[str(i)] = pfit.tolist()

        # flg, ax = plt.subplots(1)
        # for (a, b) in zip(x_total, y_total):
        #     plt.plot(a, b)
        # plt.xlabel("x[m]")
        # plt.ylabel("y[m]")
        # plt.axis("equal")
        # plt.grid(True)
        # plt.show()
        with open(f"planningData_{index}_kmh.txt", "w") as f:
            json.dump(resultInSpeed, f)
        result.append(resultInSpeed)




    with open("planningData_15_kmh.txt", "r") as f:
        data = json.load(f)
        print("Type:", type(data))
        print(data.keys())
        print(np.poly1d(np.array(data['5'])))

        # np.polyfit(x, y, 2)
    # print(x_total, y_total)
    # flg, ax = plt.subplots(1)
    # for (a, b) in zip(x_total, y_total):
    #     plt.plot(a, b)
    # plt.xlabel("x[m]")
    # plt.ylabel("y[m]")
    # plt.axis("equal")
    # plt.grid(True)

    
    

    # flg, ax = plt.subplots(1)
    # plt.plot(times, np.array(v) * 3.6)
    # plt.xlabel("Time[km/h]")
    # plt.ylabel("velocity[m]")
    # plt.grid(True)

    # #  flg, ax = plt.subplots(1)
    # #  plt.plot([math.degrees(ibeta) for ibeta in beta])
    # #  plt.grid(True)

    # plt.show()

"""


author Atsushi Sakai
"""

import math

dt = 0.1  # [s]
L = 1.5  # [m]


class State:

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, cte=0.0, epsi=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.cte = cte
        self.epsi = epsi


def update(state, a, delta, errors):

    state.x = state.x + state.v * math.cos(state.yaw) * dt
    state.y = state.y + state.v * math.sin(state.yaw) * dt
    state.yaw = state.yaw + state.v / L * delta * dt
    state.v = state.v + a * dt
    state.cte = errors[0] - state.y + state.v*math.sin(state.epsi)*dt
    state.epsi = state.yaw - errors[1] + state.v*delta/L*dt
    
    return state

def meanSquare(cte, epsi):
    # total = 0
    # for i in range(len(cte)):
    #     total += pow(cte + epsi, 2)
    # return total
    total = pow(cte + epsi, 2)
    return total

def calinput(p1, p2):
    lech = p2[0] - p1[0]
    goc = math.atan(lech/abs(p2[1] - p1[1])) * 57.2958
    return [lech, goc]

if __name__ == '__main__':
    print("start unicycle simulation")
    import matplotlib.pyplot as plt

    T = 100
    a = [1.0] * T
    delta = [math.radians(1.0)] * T
    #  print(delta)
    #  print(a, delta)

    target = [[0, 0], [1, 5], [2, 7]]

    min = 9999999
    result = []
    for i in range(-25, 25):
        state = State()
        state = update(state, 1, i, calinput(target[0], target[1]))
        state = update(state, 1, i, calinput(target[1], target[2]))
        if (meanSquare(state.cte, state.epsi) < min):
            min = meanSquare(state.cte, state.epsi)
            result = [1, i]

    print(result)
    # state = State()

    # x = []
    # y = []
    # yaw = []
    # v = []

    # for (ai, di) in zip(a, delta):
    #     state = update(state, ai, di)

    #     x.append(state.x)
    #     y.append(state.y)
    #     yaw.append(state.yaw)
    #     v.append(state.v)

    # flg, ax = plt.subplots(1)
    # plt.plot(x, y)
    # plt.axis("equal")
    # plt.grid(True)

    # flg, ax = plt.subplots(1)
    # plt.plot(v)
    # plt.grid(True)

    # plt.show()

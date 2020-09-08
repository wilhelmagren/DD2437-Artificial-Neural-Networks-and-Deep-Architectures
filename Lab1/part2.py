import numpy as np
import tensorflow
next_t = 5
beta = 0.2
gamma = 0.1
n = 10
tao = 25


def generate_input(x):
    t = np.arange([301,1500])
    input = np.array[x[t-20],x[t-15],x[t-10],x[t-5],x[t]]
    output = x[t + 5]cd
def x_t(x,t):
    res = 0
    if t - tao < 0:
        res = 0
    elif t - tao == 0:
        res = 1.5
    else:
        res = x[res]
    x[t + 1] = x[t] + (beta * res) / (1 + res ** n) - gamma * x[t]
    return x

def mackey_glass(T):
    x = []
    for t in range(T):
        denom = beta*x_t(t-tao)
        numer = 1

def split_data():
    train_x
    test_x
    train_t
    test_t
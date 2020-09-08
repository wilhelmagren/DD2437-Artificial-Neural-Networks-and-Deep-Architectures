import numpy as np
import matplotlib.pyplot as plt
#import tensorflow
next_t = 5
beta = 0.2
gamma = 0.1
n = 10
tao = 25
samples = 1200
testsamples = 200
training = 500
valid_training = 500
max = 1501
predict = 5

def generate_input(x):
    t = np.arange(301,1501)
    input = np.array([x[t-20],x[t-15],x[t-10],x[t-5],x[t]])
    output = x[t + predict]
    return input, output
def plot(x):
    y = np.linspace(-5, 1500, 1506)
    plt.plot(y,x)
    plt.show()


def mackey_glass(T):
    x = np.zeros(T + 1)
    x[0] = 1.5
    for t in range(T):
        res = t - tao
        if res < 0:
            res = 0
        elif res == 0:
            res = x[0]
        else:
            res = x[res]
        x[t + 1] = x[t] + (beta * res) / (1 + res ** n) - gamma * x[t]
    return x


x = mackey_glass(max + predict)

input, output = generate_input(x)


def split_data(input,output):
    #print(input.shape)
    train_x = np.zeros([5,training])
    test_x = np.zeros([5,testsamples])
    train_t = np.zeros([1,training])
    test_t = np.zeros([1,testsamples])
    valid_x = np.zeros([5,])
    for rows in range(len(train_x)):
        train_x[rows] = input[rows][:training]
        test_x[rows] = input[rows][training:]

    train_t = output[:training]
    test_t = output[training:]



x = mackey_glass(max + predict)
input, output = generate_input(x)
split_data(input,output)

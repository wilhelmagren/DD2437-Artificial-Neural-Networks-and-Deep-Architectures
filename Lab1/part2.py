import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

print(f"### -- Multilayer Perceptron bingbong -- ###\n\nTensorflow version: {tf.__version__}")
next_t = 5
beta = 0.2
gamma = 0.1
n = 10
tao = 25
samples = 1200
testsamples = 200
training = 500
valid_training = 1000
max = 1501
predict = 5
hidden_neurons = 2
eta = 0.001

def generate_input(x):
    t = np.arange(301,1501)
    input = np.array([x[t-20], x[t-15], x[t-10], x[t-5], x[t]])
    output = x[t + predict]
    return input, output


def plot(x):
    y = np.linspace(-5, 1500, 500)
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


def mse(X, W, T):
    return np.square(np.subtract(T, W@X)).mean()


def split_data(input, output):
    train_x = np.zeros([5, training])
    test_x = np.zeros([5, testsamples])
    train_t = np.zeros([1, training])
    test_t = np.zeros([1, testsamples])
    valid_x = np.zeros([5, training])
    valid_t = np.zeros([1, training])
    for rows in range(len(train_x)):
        train_x[rows] = input[rows][:training]
        test_x[rows] = input[rows][valid_training:]
        valid_x[rows] = input[rows][training:valid_training]

    train_t = output[:training]
    valid_t = output[training:valid_training]
    test_t = output[valid_training:]
    return train_x, train_t, valid_x, valid_t, test_x, test_t   


def model_the_fucking_data(training_data, target_data, epoch=100):
    model = keras.Sequential([
        # Flatten?
        keras.layers.Dense(5, activation='relu', name='input layer'),
        keras.layers.Dense(hidden_neurons, activation='relu', name='hidden layer'),
        keras.layers.Dense(1, name='output layer')
    ])

    # Stochastic Gradient Descent
    sgd = keras.optimizers.SGD(
        learning_rate=eta, momentum=0.0, name='SGD'
    )
    model.compile(optimizer=sgd,
                    loss=keras.losses.MeanSquaredError(),
                    metrics=['accuracy'])
    model.fit(training_data, target_data, epoch)
    test_loss, test_acc = model.evaluate(training_data, target_data, verbose=2)
    print('\nTest accuracy: ', test_acc)
    print('\nTest loss: ', test_loss)

x = mackey_glass(max + predict)
input, output = generate_input(x)
train_x, train_t, valid_x, valid_t, test_x, test_t = split_data(input, output)
model_the_fucking_data(train_x, train_t)
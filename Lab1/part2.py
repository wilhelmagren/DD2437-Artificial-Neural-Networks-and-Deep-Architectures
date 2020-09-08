import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import time

print(f"### -- Multilayer Perceptron bingbong -- ###\n\nTensorflow version: {tf.__version__}")
next_t = 5
beta = 0.2
gamma = 0.1
n = 10
tao = 25
samples = 1200
testsamples = 200
training = 1000
max = 1501
predict = 5
hidden_neurons = 8
eta = 0.001
output_nodes = 1
momen = 0.5

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

    for rows in range(len(train_x)):
        train_x[rows] = input[rows][:training]
        test_x[rows] = input[rows][training:]

    train_t = output[:training]
    test_t = output[training:]
    return train_x, train_t, test_x, test_t


def average_mse(history):
    n = len(history)
    sum = 0
    for mse in history:
        sum += mse
    return sum/n


def calculate_error(Y, T):
    return np.sum((T - Y[0]) ** 2)/len(Y[0])


def model_the_fucking_data(training_data, target_data, test_x, test_t, epoch=1000,drop_out=True):

    model = keras.Sequential()
    model.add(keras.Input(shape=(5, )))
    model.add(keras.layers.Dense(hidden_neurons))
    if(drop_out):
        model.add(keras.layers.Dropout(0.25))
    model.add(keras.layers.Dense(output_nodes))
    sgd = keras.optimizers.SGD(learning_rate=eta, momentum=momen)
    early_ritsch = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    model.compile(optimizer=sgd,
                    loss='mse')
    start_t = time.time()
    hist = model.fit(training_data, target_data, epochs=epoch, batch_size=32, verbose=1, validation_split=0.2, callbacks=[early_ritsch])
    end_t = time.time()
    print('time tooked: ', end_t - start_t)
    prediction_van_darkholme = model.predict(test_x)
    # i van darkholme har vi alla predictions. sen gör vi MSE med predictions och test_t DETTA GER OSS MSE för test
    #
    #
    #
    #
    # TO DO:
    # AVERGAE MSE FOR HIST-HISTORY otherwise we just check smallest. FOR PREDICTION WE DO MSE OVER ALL THINGS SO IT IS LIKE AVERAGE

    print(f"Training MSE: {average_mse(hist.history['val_loss'])}")
    print(f"Testing MSE: {calculate_error(prediction_van_darkholme.T, test_t)}")
    #weight, bias = model.layers[0].get_weights()
    plt.title('Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Prediction')
    plt.plot(prediction_van_darkholme, label='prediction')
    plt.plot(test_t, label='TARGET')
    plt.legend()
    plt.show()
    #return weight

x = mackey_glass(max + predict)
input, output = generate_input(x)
train_x, train_t, test_x, test_t = split_data(input, output)
weight = model_the_fucking_data(train_x.T, train_t, test_x.T, test_t)
plt.hist(weight, bins='auto', label='weight histogram')
plt.xlabel("weight value")
plt.ylabel("number of weights")
plt.show()
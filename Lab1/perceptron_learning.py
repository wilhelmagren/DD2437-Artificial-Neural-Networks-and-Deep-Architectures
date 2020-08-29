import numpy as np
import matplotlib.pyplot as plt

n = 100
mA = [-1.0, 1.0]
mB = [-1.0, 1.0]
sigmaA = 0.5
sigmaB = 0.5
use_bias = True

def generate_matrices():
    # Bias is the third row in X.
    #    |-> Dimensions are: 3x2n
    X = np.ones([3, 2*n])

    # Randomize set_1 both x & y data
    X[0, :n] = np.random.rand(1, n) * sigmaA + mA[0]
    X[0, n:] = np.random.rand(1, n) * sigmaA + mA[1]

    # Randomize set_2 both x & y data
    X[1, :n] = np.random.rand(1, n) * sigmaB + mB[0]
    X[1, n:] = np.random.rand(1, n) * sigmaB + mB[1]

    # Set the bias
    X[2, :2*n] = 1 if use_bias else 0


    # Weight matrix generation works (Also with bias)
    # X.shape[0] gives it the same dimensions as the number of rows in X
    #     |-> the dimensions are 1x3
    W = np.array([np.random.normal(0, 1, X.shape[0])])

    # Placing the bias in the last spot of the weight matrix W yields the correct delta_W
    W[0][2] = 1.0

    # Target matrix generation is correct.
    #     |-> the dimensions are 1x2n
    T = np.array([np.concatenate([[-1]*n, [1]*n])])

    # Plot the generated data sets found in X
    #plot_sets(X)
    return X, W, T

def plot_sets(X):
    plt.scatter(X[0, n:], X[1, n:], color="red")
    plt.scatter(X[0, :n], X[1, :n], color="blue")
    plt.title("Perceptron learning plot")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def plot_all(X, W):
    plt.scatter(X[0, n:], X[1, n:], color="red")
    plt.scatter(X[0, :n], X[1, :n], color="blue")
    plt.title("Perceptron learning plot")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(top = 3, bottom = -3)
    x = np.linspace(-3, 3, 200)
    bias = W[0][2]
    k = -(bias/W[0][1])/(bias/W[0][0])
    m = -bias/W[0][1]
    y = k*x + m
    plt.plot(x, y, color="green")
    plt.show()


def perceptron_rule(X, E, eta):
    return eta*(E@np.transpose(X))


def perceptron_learning(X, W, T, num_epoch):
    E = np.zeros([1, 2*n])
    Y = np.zeros([1, 2*n])
    for i in range(num_epoch):
        # W has dimensions: 1x3, X has dimensions: 3x2n
        #     |-> Y-prime will get dimensions: 1x2n
        Y_prime = W@X
        for j in range(2*n):
            if Y_prime[0][j] <= -1:
                Y[0][j] = -1
            else:
                Y[0][j] = 1
            E[0][j] = T[0][j] - Y[0][j]
        delta_W = perceptron_rule(X, E, 0.001)
        W = delta_W + W
    return W


def main():
    input_x, weight, target = generate_matrices()
    plot_all(input_x, weight)
    new_weight = perceptron_learning(input_x, weight, target, 10000)
    plot_all(input_x, new_weight)


if __name__ == "__main__":
    main()
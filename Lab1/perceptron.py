import numpy as np
import matplotlib.pyplot as plt
import math
n = 100
mA = [-0.5, 0.0]
mB = [-0.5, 0.0]
#if we want to shuffle set it to 1
shuffle = 0
bias = True
sigmaA = 0.5
sigmaB = 0.5

"""
    Returns three numpy matrixes -
        X, W, T
"""
def generate_matrices():
    # set1
    set_a1 = np.random.rand(1,n) * sigmaA + mA[0]
    set_a2 = np.random.rand(1,n) * sigmaA + mA[1]
    if shuffle == 1 :
        np.random.shuffle(set_a1)
        np.random.shuffle(set_a2)

    # set2
    set_b1 = np.random.rand(1, n) * sigmaB + mB[0]
    set_b2 = np.random.rand(1, n) * sigmaB + mB[1]

    if shuffle == 1 :
        np.random.shuffle(set_b1)
        np.random.shuffle(set_b2)
    if bias:
        X = np.array([np.concatenate((set_a1, set_a2)), np.concatenate((set_b1, set_b2)), np.array([1] * (2*n))])
    else:
        X = np.array([np.concatenate((set_a1, set_a2)), np.concatenate((set_b1, set_b2))])

    W = np.random.rand(1,3)
    W[0][0] = 1
    T = np.concatenate([[-1]*n, [1]*n])
    return X,W,T
def perceptron_rule(eta,e,X):
    return eta*(e@np.transpose(X))
def perceptron_learning(X,W,T,epoch):
    e = np.zeros(2*n)
    Y = np.zeros(2*n)
    for i in range(epoch):
        y_prime = W@X
        for j in range(2*n):
            if y_prime[j] <= -1:
                Y[j] = -1
            else:
                Y[j] = 1
            e[j] = T[j] - Y[j]
        deltaW = perceptron_rule(0.001,e,X)
        W = deltaW + W
        print(W)
        print(Y)
def plot(X):
    plt.scatter(X[0][0],X[1][0])
    plt.scatter(X[0][1],X[1][1])
    plt.show()
X,W,T = generate_matrices()
print(X.shape)
print(W.shape)
perceptron_learning(X,W,T,2000)

#plot(X)




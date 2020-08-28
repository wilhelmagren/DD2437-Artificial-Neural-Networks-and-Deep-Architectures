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

    input_matrix_X = []

    input_matrix_X = np.array([np.concatenate((set_a1, set_a2)), np.concatenate((set_b1, set_b2)), np.array([1] * (2*n)),np.array([0] * (2*n))])

    #input_matrix_X = np.array([np.concatenate([set_a1, set_a2]), np.concatenate([set_b1, set_b2])])
    weight_matrix_W = np.random.rand(1,2*n)
    target_matrix_T = np.concatenate([[0.5]*n, [-0.5]*n])
    return input_matrix_X, weight_matrix_W, target_matrix_T

def delta_rule(X,W,t,eta = 0.001):
    return -eta*(W@X-t)@np.transpose(X)

def perceptron_rule(eta,e,X):
    return eta*(e@np.transpose(X))
def perceptron_learning(X,W,T,epoch):
    e = 0
    output = []
    for i in range(epoch):
        y_predict = W@X
        for i in range(n*2):
            if y_predict[i] > 0:
                output[i] = 1
            else:
                output[i] = 0
        #Train weights using delta rule
        e = T[i] - y_predict[i]
        deltaW = perceptron_rule(0.001,e,X)
        W = W + deltaW
        print(W)




"""
Plot the input matrix
"""
def plot(X):
    plt.scatter(X[0][0],X[1][0])
    plt.scatter(X[0][1],X[1][1])
    plt.show()
"""
Find difference of numbers
"""
def square_sum(prev_val, new_val):
    return math.sqrt((prev_val - new_val)^2)

"""
If the new val is really close to the previous val - we can say we have converged #BROSCIENCE
"""
def convergence_check(prev_val, new_val):
    if square_sum(prev_val, new_val) <= 0.00001 :
        return True
    return False

W = np.array([1, 0.5 ,0.1])
X = np.array([[-1,1,-1,1],[-1,-1,1,1],[1,1,1,1]])
T = np.array([-1,-1,-1,1])
e = np.array([0,0,0,0])
for i in range(2000):
    output = np.array([0, 0, 0, 0])
    Y = W @ X
    for j in range(4):
        if Y[j] <= -1:
            output[j] = -1
        else:
            output[j] = 1
        e[j] = T[j] - output[j]
    deltaW = perceptron_rule(0.01,e,X)
    W = deltaW + W
    print(output)
    print(W)



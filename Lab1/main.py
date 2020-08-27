import numpy as np
import matplotlib.pyplot as plt
import math
n = 100
mA = [0.0, 0.5]
mB = [-0.5, 0.0]
#if we want to shuffle set it to 1
shuffle = 0
bias = 1
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
    bias_array = []
    if bias == 1 :
        bias_array = np.ones(n)
    if shuffle == 1 :
        np.random.shuffle(set_a1)
        np.random.shuffle(set_a2)

    # set2
    set_b1 = np.random.rand(1, n) * sigmaB + mB[0]
    set_b2 = np.random.rand(1, n) * sigmaB + mB[1]
    if shuffle == 1 :
        np.random.shuffle(set_b1)
        np.random.shuffle(set_b2)

    input_matrix_X = np.array([np.concatenate(np.concatenate(set_a1, set_a2), np.concatenate(set_b1, set_b2))])
    weight_matrix_W = np.random.rand(1,n)
    target_matrix_T = np.concatenate([0.5]*n, [-0.5]*n)
    return input_matrix_X, weight_matrix_W, target_matrix_T

def delta_rule(X,W,t,lr = 0.001):
    return -lr*(W@X-t)@np.transpose(X)

# Variable to see if we have converged
convergence = 0
def delta_train(X, W, T, epoch):
    #Iterate epoch times
    for i in (0, epoch):
        delta = delta_rule(X,W,T)
        W = W + delta

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

X, W, T = generate_matrices()

delta_train(X,W,T,100)

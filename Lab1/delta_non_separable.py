import perceptron_and_delta as pad
import numpy as np
import matplotlib.pyplot as plt
import random

use_bias = True
ndata = 100
n = 100
def generate_overlap_matrices():

    mA = [-0.5, 0.3]
    mB = [1.0, 0.0]
    sigmaA = 2
    sigmaB = 2

    """
    Func generate_matrices/0
    @spec generate_matrices() :: np.array(), np.array(), np.array()

        Generates three randomized numpy arrays.
            |-> An input array X with dimensions: 3x2n
            |-> A weight array W with dimensions: 1x3
            |-> A target array T with dimensions: 1x2n
    """
    # Bias is the third row in X.
    #    |-> Dimensions are: 3x2n
    X = []
    if use_bias:
        X = np.ones([3, 2*n])
        X[2, :2 * n] = 1
    else:
        X = np.ones([2,2*n])

    # Randomize both set_1 and 2 x data
    X[0, :n] = np.random.rand(1, n) * sigmaA + mA[0]
    X[0, n:] = np.random.rand(1, n) * sigmaA + mA[1]

    # Randomize both set_1 and 2 y data
    X[1, :n] = np.random.rand(1, n) * sigmaB + mB[0]
    X[1, n:] = np.random.rand(1, n) * sigmaB + mB[1]




    # Weight matrix generation works (Also with bias)
    # X.shape[0] gives it the same dimensions as the number of rows in X
    #     |-> the dimensions are 1x3
    W = np.array([np.random.normal(0, 1, X.shape[0])])

    # Placing the bias in the last spot of the weight matrix W yields the correct delta_W
    if use_bias:
        W[0][2] = 1.0

    # Target matrix generation is correct.
    #     |-> the dimensions are 1x2n
    T = np.array([np.concatenate([[-1]*n, [1]*n])])

    # Plot the generated data sets found in X
    return X, W, T



def generate_matrices_25():

    mA = [1.0, 0.3]
    mB = [0.0, 0.0]
    sigmaA = 0.2
    sigmaB = 0.3
    X = np.zeros([3, ndata * 2])
    nhalf = int(ndata / 2)
    """splits data in to non linearly sets
    First row = """

    X[0,:nhalf] = np.random.rand(1,round(0.5*ndata)) * sigmaA - mA[0]
    X[0,nhalf:ndata] = np.random.rand(1,round(0.5*ndata)) * sigmaA + mA[0]

    X[0,ndata:2*ndata] = np.random.rand(1,ndata) * sigmaB + mB[0];

    X[1, :ndata] = np.random.rand(1, ndata) * sigmaB + mB[1]
    X[1, ndata:] = np.random.rand(1, ndata) * sigmaB + mA[1]
    X[2,:2*ndata] = 1

    # Weight matrix generation works (Also with bias)
    # X.shape[0] gives it the same dimensions as the number of rows in X
    #     |-> the dimensions are 1x3
    W = np.array([np.random.normal(0, 1, X.shape[0])])

    # Placing the bias in the last spot of the weight matrix W yields the correct delta_W
    if use_bias:
        W[0][2] = 1.0

    # Target matrix generation is correct.
    #     |-> the dimensions are 1x2n
    T = np.ones([1, 150])
    T[0, :75] = -1
    index_i = 0
    index_j = 0
    rand_matrix_25 = np.ones([3,150])
    count = 0
    for i in range(0,75):
        if random.randint(0,1) == 0:
            count += 1
    if count < 25 :
        count = 25
    if count > 50:
        count = 50
    otherpart = 75-count

    rand_matrix_25[0, :count] = X[0, :count]
    otherpart_XD = otherpart + 50
    rand_matrix_25[0, count:75] = X[0, 50:otherpart_XD]
    rand_matrix_25[1, :count] = X[1, :count]
    rand_matrix_25[1, count:75] = X[1, 50:otherpart_XD]

    rand_matrix_25[0, 75:] = X[0, 100:175]
    rand_matrix_25[1, 75:] = X[1, 100:175]
    # Plot the generated data sets found in X
    return rand_matrix_25, W, T

def generate_matrices_50_BLUEDABADEBA():

    mA = [1.0, 0.3]
    mB = [0.0, 0.0]
    sigmaA = 0.2
    sigmaB = 0.3
    X = np.zeros([3, ndata * 2])
    nhalf = int(ndata / 2)
    """splits data in to non linearly sets
    First row = """

    X[0,:nhalf] = np.random.rand(1,round(0.5*ndata)) * sigmaA - mA[0]
    X[0,nhalf:ndata] = np.random.rand(1,round(0.5*ndata)) * sigmaA + mA[0]

    X[0,ndata:2*ndata] = np.random.rand(1,ndata) * sigmaB + mB[0];

    X[1, :ndata] = np.random.rand(1, ndata) * sigmaB + mB[1]
    X[1, ndata:] = np.random.rand(1, ndata) * sigmaB + mA[1]
    X[2,:2*ndata] = 1

    # Weight matrix generation works (Also with bias)
    # X.shape[0] gives it the same dimensions as the number of rows in X
    #     |-> the dimensions are 1x3
    W = np.array([np.random.normal(0, 1, X.shape[0])])

    # Placing the bias in the last spot of the weight matrix W yields the correct delta_W
    if use_bias:
        W[0][2] = 1.0

    # Target matrix generation is correct.
    #     |-> the dimensions are 1x2n
    T = np.ones([1, 150])
    T[0, :75] = -1
    index_i = 0
    index_j = 0
    rand_matrix_25 = np.ones([3,150])
    count = 0
    for i in range(0,50):
        if random.randint(0,1) == 0:
            count += 1
    otherpart = 50-count

    rand_matrix_25[0, :count] = X[0, :count]
    otherpart_XD = otherpart + 50
    rand_matrix_25[0, count:50] = X[0, 50:otherpart_XD]
    rand_matrix_25[1, :count] = X[1, :count]
    rand_matrix_25[1, count:50] = X[1, 50:otherpart_XD]

    rand_matrix_25[0, 50:] = X[0, 100:200]
    rand_matrix_25[1, 50:] = X[1, 100:200]
    # Plot the generated data sets found in X
    return rand_matrix_25, W, T

def generate_matrices_50_redd():

    mA = [1.0, 0.3]
    mB = [0.0, 0.0]
    sigmaA = 0.2
    sigmaB = 0.3
    X = np.zeros([3, ndata * 2])
    nhalf = int(ndata / 2)
    """splits data in to non linearly sets
    First row = """

    X[0,:nhalf] = np.random.rand(1,round(0.5*ndata)) * sigmaA - mA[0]
    X[0,nhalf:ndata] = np.random.rand(1,round(0.5*ndata)) * sigmaA + mA[0]

    X[0,ndata:2*ndata] = np.random.rand(1,ndata) * sigmaB + mB[0];

    X[1, :ndata] = np.random.rand(1, ndata) * sigmaB + mB[1]
    X[1, ndata:] = np.random.rand(1, ndata) * sigmaB + mA[1]
    X[2,:2*ndata] = 1

    # Weight matrix generation works (Also with bias)
    # X.shape[0] gives it the same dimensions as the number of rows in X
    #     |-> the dimensions are 1x3
    W = np.array([np.random.normal(0, 1, X.shape[0])])

    # Placing the bias in the last spot of the weight matrix W yields the correct delta_W
    if use_bias:
        W[0][2] = 1.0

    # Target matrix generation is correct.
    #     |-> the dimensions are 1x2n
    T = np.ones([1, 150])
    T[0, :75] = -1
    index_i = 0
    index_j = 0
    rand_matrix_25 = np.ones([3,150])

    rand_matrix_25[0, :100] = X[0, :100]
    rand_matrix_25[1, :100] = X[1, :100]

    rand_matrix_25[0, 100:] = X[0, 100:150]
    rand_matrix_25[1, 100:] = X[1, 100:150]
    # Plot the generated data sets found in X
    return rand_matrix_25, W, T


def main():
    learning_rate = 0.001
    X, W, T = generate_matrices_25()
    print("err.str\n    |-> performing perceptron learning...")
    perceptron_weight, perceptron_acc = pad.perform_perceptron(X, W, T, learning_rate, True)
    print("err.str\n    |-> performing delta batch learning...")
    delta_weight, error_list, delta_acc = pad.perform_delta(X, W, T, learning_rate, True, True)
    # perceptron_learning will be red in below plot and delta rule will be blue
    pad.plot_diff(perceptron_acc, delta_acc)
    # print("err.str\n    |-> performing delta sequential learning...")
    # perform_delta(X, W, T, learning_rate, False)
    exit()


if __name__ == "__main__":
    main()

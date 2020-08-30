import numpy as np
import matplotlib.pyplot as plt

"""
Apply and compare perceptron learning with the Delta learning rule in 
batch mode on the generated dataset. Adjust the learning rate and study 
the convergence of the two algorithms.

Last updated: 30/08/2020

Assignment - Part 1
    1.  Apply and compare perceptron learning with the Delta learning rule in batch mode.
        Adjust the learning rate and study the convergence of the two algorithms.
    
    2.  Compare sequential to batch learning approach for the Delta rule. 
        How quickly (in terms of epochs) do the algorithm converge? 
    
    3. Remove the bias, train your network with the Delta rule in batch mode and test its behaviour.
"""
n = 100
mA = [-1.0, 1.0]
mB = [-1.0, 1.0]
sigmaA = 0.5
sigmaB = 0.5
use_bias = True


def generate_matrices():
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
    return X, W, T


def plot_sets(X):
    """
    Func plot_sets/1
    @spec plot_sets(np.array()) :: void

        Plots the generated datasets according to matrix
        generation in generate_matrices/0.
    """
    plt.scatter(X[0, n:], X[1, n:], color="red")
    plt.scatter(X[0, :n], X[1, :n], color="blue")
    plt.title("Perceptron learning plot")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def plot_all(X, W, do_delta, eta=0.001, iteration=0):
    """
    Func plot_all/2
    @spec plot_all(np.array(), np.array()) :: void

        Plots both the perceptron line & both datasets.
        We can find the line due to the following property:
            Wx = 0
            which in our case means: w0 + w1x1 + w2x2 = 0
    """
    plt.scatter(X[0, n:], X[1, n:], color="red")
    plt.scatter(X[0, :n], X[1, :n], color="blue")
    if do_delta:
        plt.title(f"Delta rule: eta = {eta}, iteration = {iteration}")
    else:
        plt.title(f"Perceptron learning rule: eta = {eta}, iteration = {iteration}")
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


def check_convergence(old_error, new_error, diff_threshold=0.001):
    """
    Func check_convergence/3
    @spec check_convergence(np.array(), np.array(), integer) :: boolean
        Calculates the quota between old and new error and compares to arbitrary percentage.
        Convergence theorem:
            ,,If a solution exists for a finite training dataset when perceptron learning
                always converges after a finite number of sets (independent of eta)"
    """
    if(np.abs(old_error - new_error)/old_error) <= diff_threshold:
        return True
    return False


def calculate_error(X, W, T):
    """
    Func calculate error/3
    @spec calculate_error(np.array(), np.array(), np.array()) :: integer
        Calculates the difference between our estimate (W@X) and the target value.
        Using np.sum/1 to sum the matrix and get a numerical value.
    """
    return np.sum((T - W@X) ** 2)


def delta_rule(X, W, T, eta):
    """
    Func delta_rule/4
    @spec delta_rule(np.array(), np.array(), np.array(), integer) :: np.array()
        Calculates the values that we need to update our weight matrix W with.
        Returns a matrix with dimensions the same as the weight matrix W.
        See lab1.pdf for formula.
    """
    return -eta*(W@X - T)@np.transpose(X)


def delta_learning(X, W, T, eta):
    """
    Func delta_learning/3
    @spec delta_learning(np.array(), np.array(), np.array()) :: np.array()
        Iteratively calculates the delta_rule on the dataset X, weight matrix W and target matrix T.
        Function will return when the error between W and deltaW converges.
        (See check_convergence/3 for logical expression of convergence)
    """
    converged = False
    iteration = 0
    while not converged:
        # Plot the perceptron line after each 2 iteration
        if (iteration % 2) == 0:
            plot_all(X, W, True, eta, iteration)

        prev_W = W
        delta_W = delta_rule(X, W, T, eta)
        W = delta_W + prev_W
        prev_error = calculate_error(X, prev_W, T)
        new_error = calculate_error(X, W, T)
        if check_convergence(prev_error, new_error):
            return W, iteration
        iteration += 1


def perceptron_rule(X, E, eta):
    """
    Func perceptron_rule/3
    @spec perceptron_rule(np.array(), np.array(), integer) :: np.array()
        Performs a matrix multiplication with E & X -> then performs scalar multiplication with eta.

        E.shape() :: NxK
        X.shape() :: MxK
        The returned matrix will according to matrix multiplication property have the shape NxM, since we take X^t
    """
    return eta*(E@np.transpose(X))


def perceptron_learning(X, W, T, eta, num_epoch):
    """
    Func perceptron_learning/4
    @spec perceptron_learning(np.array(), np.array(), np.array(), integer) :: np.array()
        Perform epoch number of perceptron learning iterations in batch mode. Function
        returns the updated weight matrix when network function converges.

    """
    E = np.zeros([1, 2*n])
    Y = np.zeros([1, 2*n])
    for i in range(num_epoch):
        # Plot the perceptron line after each 2 iteration
        if (i % 2) == 0:
            plot_all(X, W, False, eta, i)

        # W has dimensions: 1x3, X has dimensions: 3x2n
        #     |-> Y-prime will get dimensions: 1x2n
        Y_prime = W@X
        for j in range(2*n):
            if Y_prime[0][j] <= -1:
                Y[0][j] = -1
            else:
                Y[0][j] = 1
            E[0][j] = T[0][j] - Y[0][j]
        delta_W = perceptron_rule(X, E, eta)
        W = delta_W + W
    return W


def perform_perceptron(eta):
    """
    Func perform_perceptron/0
    @spec perform_perceptron() :: void
        Trains the perceptron using perceptron learning rule.
        Number of epochs is equal to: ${see_below}
    """
    input_x, weight, target = generate_matrices()
    plot_all(input_x, weight, False, eta)
    num_epochs = 100
    print(f"    |-> starting training with {num_epochs} number of epochs...")
    new_weight = perceptron_learning(input_x, weight, target, eta, num_epochs)
    print("    |-> training done.")
    plot_all(input_x, new_weight, False, eta, num_epochs)


def perform_delta(eta):
    """
    Func perform_delta/0
    @spec perform_delta() :: void
        Trains the perceptron the delta rule.
        Terminates when the error converges.
    """
    input_x, weight, target = generate_matrices()
    plot_all(input_x, weight, True, eta)
    print("    |-> starting training...")
    new_weight, number_of_iterations = delta_learning(input_x, weight, target, eta)
    print("    |-> training done.")
    plot_all(input_x, new_weight, True, eta, number_of_iterations)


def main(do_delta):
    learning_rate = 0.001
    if not do_delta:
        print("err.str\n    |-> performing perceptron learning...")
        perform_perceptron(learning_rate)
    else:
        print("err.str\n    |-> performing delta learning...")
        perform_delta(learning_rate)


if __name__ == "__main__":
    main(True)

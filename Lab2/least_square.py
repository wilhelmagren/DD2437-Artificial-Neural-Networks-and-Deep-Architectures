import numpy as np
import matplotlib.pyplot as plt
"""
 Use Gaussian RBF's to approximate simple functions of one variable.
    Every unit in the hidden layer implement the transfer function phi_i(x)
 The output layer calculates the weighted sum of the n hidden layer units.
 
 (1)   f^(x) = sum from i to n Wi * phi_i(x)
 
 The RBF layer maps the input space to an n-dimensional vector.
 n is usually higher than the dimension of the input space.
 
 We want to find weights which minimize the total approximation error summed 
 over all N patterns used as training examples.
 
 f(xk) => f() is the target function, xk is the k'th pattern, 
 we write a linear equation system with ONE ROW PER PATTERN, where each
 row states the above equation (1) for a particular pattern.
 
 If the number of inputs are greater than the number of units in the hidden layer: N > n
 the system is overdetermined - cannot use Gaussian elimination to solve for W.
 
 Reflect over the questions:
    [] What is the lower bound for the number of training examples, N?
    [] What happens with the error if N = n? Why?
    [] Under what conditions, if any, does the system of linear equations have a solution?
    [] During training we use an error measure defined over the training examples. 
       Is it good to use this measure when evaluating the performance of the network? Explain!

 Two different methods for determining the weights w_i:
    batch mode using least squares
    sequential (instrumental, on-line) learning using the delta rule
    
        We can write the system of linear equations as: PHI * w = f

                    ----------------------------------------
                    | phi_1(x1), phi_2(x1), ..., phi_n(x1) |
                    | phi_1(x2), phi_2(x2), ..., phi_n(x2) |
        where PHI = | phi_1(x3), phi_2(x3), ..., phi_n(x3) |
                    | .                                    |
                    | phi_1(xn), phi_2(xn), ..., phi_n(xn) |
                    ----------------------------------------

        and W = [w_1, w_2, ..., w_n]^T

        The function error becomes:
            total error = ||PHI*w - f||^2

        According to linear algebra - obtain W by solving the following system:
            PHI^T * PHI * W = PHI^T * f
"""

# Global variables don't touch please
N = 63  # Number of inputs
n = 25  # Number of RBF's, has to be greater than N
step_size = 0.1  # Used for generating sin wave
sigma = 0.5  # Variance for all nodes


def generate_input(use_noise=True):
    """
    Func generate_input/0
    @spec generate_input() :: np.array(), np.array(), np.array(), np.array(), np.array(), np.array()
        Generate all necessary training, testing and target data used in the training and validation process.
    """
    x_training = np.arange(0, 2 * np.pi, step_size)
    x_testing = np.arange(0.05, 2 * np.pi + 0.05, step_size)
    training_target_sin = np.sin(2 * x_training)
    testing_target_sin = np.sin(2 * x_testing)
    square_training_target = np.sign(training_target_sin)
    square_testing_target = np.sign(testing_target_sin)
    if use_noise:
        x_training += np.random.normal(0, 0.1, x_training.shape)
        x_testing += np.random.normal(0, 0.1, x_testing.shape)
    for i in range(len(square_training_target)):
        if square_training_target[i] == 0:
            square_training_target[i] = 1
    for i in range(len(square_testing_target)):
        if square_testing_target[i] == 0:
            square_testing_target[i] = 1

    return x_training, x_testing, training_target_sin, testing_target_sin, square_training_target, square_testing_target


def phi_func(x, my):
    return np.exp(-(x - my)**2 / (2*(sigma ** 2)))


def calculate_error(target,estimate):
    return 1/2*((target-estimate)**2)


def delta_learning_rule(error, phi, k, eta):
    return eta*error*phi[k]


def delta_rule(square):
    train_x, test_x, sin_train_t, sin_test_t, square_train_t, square_test_t = generate_input()
    w = generate_weight()
    rbf_pos = generate_initial_rbf_position()
    phi_test = generate_big_phi(test_x, rbf_pos)
    phi_train = generate_big_phi(train_x, rbf_pos)
    epochs = 1000
    train_size = len(train_x)
    error = 0
    estimation = []
    for i in range(epochs):
        print(f" Epoch number: [{i}]")
        if not square:
            for k in range(train_size):
                # print(f" k number: [{k}]")
                e = sin_train_t[k] - phi_train[k]@w
                # k = (k+1) % train_size
                delta_w = delta_learning_rule(e, phi_test, k, 0.001)
                w += delta_w
        else:
            for k in range(train_size):
                # print(f" k number: [{k}]")
                e = square_train_t[k] - phi_train[k] @ w
                # k = (k + 1) % train_size
                delta_w = delta_learning_rule(e, phi_test, k, 0.001)
                w += delta_w
        if not square:
            estimation = phi_test @ w
            error = np.abs(estimation - sin_test_t).mean()
            print(f"Epoch number [{i}] has error: [{error}]")
        else:
            estimation = phi_test @ w
            for j in range(len(estimation)):
                if estimation[j] >= 0:
                    estimation[j] = 1
                else:
                    estimation[j] = -1
            error = np.abs(estimation - square_test_t).mean()
            print(f"Epoch number [{i}] has error: [{error}]")
    """if not square:
        estimate = phi_test @ w
        error = np.abs(estimate-sin_test_t).mean()
    else:
        estimation = phi_test @ w
        for i in range(len(estimation)):
            if estimation[i] >= 0:
                estimation[i] = 1
            else:
                estimation[i] = -1
        error = np.abs(estimation-square_test_t).mean()"""
    return sin_test_t, estimation


def generate_big_phi(input_matrix, rbf_pos):
    """
    Func generate_big_phi/2
    @spec generate_big_phi(np.array(), np.array()) :: np.array()
        Calculates and generates the Gaussian RBF phi matrix based on the transfer function given in the lab instructions.
    """
    big_phi_matrix = np.zeros([N, n])
    for i in range(N):
        for j in range(n):
            big_phi_matrix[i][j] = phi_func(input_matrix[i], rbf_pos[j])
    return big_phi_matrix


def generate_weight():
    """
    Func generate_weight/0
    @spec generate_weight() :: np.array()
        Generates a gaussian distributed numpy array with weight values.
    """
    return np.random.uniform(0, 2*np.pi, n)


def generate_initial_rbf_position():
    """
    Func generate_initial_rbf_position/0
    @spec generate_initial_rbf_position() :: np.array()
        Generate a uniformed list of starting positions for the RBF's

        TODO: Find out how to most accurately position the RBF's to predict the target function
    """
    rbf_pos = np.random.uniform(0, 2*np.pi, n)
    return rbf_pos


def calc_total_error(estimation, target):
    """
    The function error becomes:
            total error = ||PHI*w - f||^2
    """
    return np.abs(np.subtract(estimation, target)).mean()


def least_squares(phi, target):
    """
    Func least_squares/2
    @spec least_squares(np.array(), np.array()) :: np.array()
        Calculates the weight matrix w by solving the given systems of linear equations.
    """
    return np.linalg.solve(phi.T @ phi, phi.T @ target)


def plot_approximation(estimate, target):
    """
    Func plot_approximation/2
    @spec plot_approximation(list, list) :: void
        Plot the function approximation estimate over the target function.
    """
    print(estimate)
    plt.plot(estimate, label="Estimate", color="red")
    plt.plot(target, label="Target", color="blue")
    plt.title("Estimate vs target")
    plt.grid()
    plt.legend()
    plt.show()


def plot_error(err, it):
    """
    Func plot_error/2
    @spec plot_error(list, list) :: void
        Plots the calculated errors over the corresponding iteration in learning process.
        Used to visualize the convergence of the error function E.
    """
    plt.ylim(top=-0.1, bottom=1)
    # plt.plot(x, y, ...)
    plt.plot(it, err, color="green")
    plt.xlabel("Number of units")
    plt.ylabel("Error")
    plt.title("Error ratio over number units")
    plt.grid()
    plt.show()


def perform_least_squared(squared=False):
    """
    Func perform_least_squared/0
        Change code to work for 'box'-function instead of sinus.
    """
    print("### --- Doing the least squared --- ###")
    train_x, test_x, sin_train_t, sin_test_t, square_train_t, square_test_t = generate_input()
    err_list = []
    iteration_list = []
    iteration = 0
    global n
    for q in range(100):
        if not squared:
            rbf_pos = generate_initial_rbf_position()
            big_phi = generate_big_phi(train_x, rbf_pos)
            w = least_squares(big_phi, sin_train_t)
            test_phi = generate_big_phi(test_x, rbf_pos)
            estimation = test_phi @ w
            err_list.append(calc_total_error(estimation, sin_test_t))
            print(err_list[iteration])
            iteration_list.append(iteration + n)
            iteration += 1
            if err_list[iteration - 1] <= 0.001:
                print(f"Number of RBF's: [{n}] and error [0.001]")
                return estimation, sin_test_t, err_list, iteration_list, rbf_pos
            n += 1
        else:
            print(iteration)
            print(n)
            rbf_pos = generate_initial_rbf_position()
            big_phi = generate_big_phi(train_x, rbf_pos)
            w = least_squares(big_phi, square_train_t)
            test_phi = generate_big_phi(test_x, rbf_pos)
            estimation = test_phi @ w
            for i in range(len(estimation)):
                if estimation[i] >= 0:
                    estimation[i] = 1
                else:
                    estimation[i] = -1
            err_list.append(calc_total_error(estimation, square_test_t))
            print(err_list[iteration])
            iteration_list.append(iteration + n)
            iteration += 1
            if err_list[iteration - 1] <= 0.01:
                print(f"Number of RBF's: [{n}]")
                return estimation, square_test_t, err_list, iteration_list, rbf_pos
            n += 1


def plot_rbf_pos(rbf):
    """
    Func plot_rbf_pos/1
    @spec plot_approximation(list) :: void
        Plot the RBF positions in the x-y plane
    """
    for i in range(len(rbf)):
        plt.scatter(rbf[i][0], rbf[i][1], color="red")
    plt.title("RBF positions")
    plt.grid()
    plt.legend()
    plt.show()


def main():
    target, est = delta_rule(False)
    plot_approximation(est, target)
    # est, tar, error_lists, it_list, rbf_positions = perform_least_squared(True)
    # plot_error(error_lists, it_list)
    # plot_rbf_pos(rbf_positions)
    # plot_approximation(est, tar)


if __name__ == "__main__":
    main()

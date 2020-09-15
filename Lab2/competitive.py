import numpy as np
import matplotlib.pyplot as plt
import random

# Global variables don't touch please
N = 63  # Number of inputs
n = 8  # Number of RBF's, has to be greater than N
step_size = 0.1  # Used for generating sin wave
sigma = 0.5  # Variance for all nodes


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


def plot_scatter_stuff(patterns, est, target):
    plt.scatter(patterns, target, color="red", label="Correct")
    plt.scatter(est, target, color="blue", label="Estimation")
    plt.grid()
    plt.legend()
    plt.show()


def generate_input(use_noise=False):
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


def delta_learning_rule(error, phi, k, eta):
    return eta*error*phi[k]


def calculate_error(target, estimate):
    return 1/2*((target-estimate)**2)


def phi_func(x, my):
    return np.exp(-(x - my)**2 / (2*(sigma ** 2)))


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


def place_rbf_hand_job():
    return np.array([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4])


def generate_initial_rbf_position():
    """
    Func generate_initial_rbf_position/0
    @spec generate_initial_rbf_position() :: np.array()
        Generate a uniformed list of starting positions for the RBF's

        TODO: Find out how to most accurately position the RBF's to predict the target function
    """
    rbf_pos = np.random.uniform(0, 2*np.pi, n)
    return rbf_pos


def find_closest_rbf(x, rbf, w, eta):
    # Find what weight minimizes the distance x - w
    d = 10000
    index_count = 0
    min_dist_index = 0
    for w_i in w:
        distance = np.linalg.norm(x - w_i)
        if distance < d:
            min_dist_index = index_count
            d = distance
        index_count += 1

    # We have what W_i is closest to x, so RBF_i is closest
    rbf[min_dist_index] += eta * (x - rbf[min_dist_index])
    return rbf


def delta_rule(x_pattern, w, rbf_pos, epochs=10000, eta=0.01):
    error_list = []
    iteration_list = []
    M = len(x_pattern)
    for i in range(epochs):
        # Pick random pattern from X
        # find which rbf pos is closest to randomized X
        for k in range(M):
            rand_index = k  # random.randint(0, N - 1)
            rand_x = x_pattern[rand_index]
            rbf_pos = find_closest_rbf(rand_x, rbf_pos, w, eta)
    return rbf_pos


def do_competitive_learning_():
    x_training, x_testing, training_target_sin, testing_target_sin, square_training_target, square_testing_target = generate_input()
    w = generate_weight()
    # rbf_pos = place_rbf_hand_job()
    rbf_pos = generate_initial_rbf_position()
    rbf_pos = delta_rule(x_training, w, rbf_pos)
    estimation = generate_big_phi(x_training, rbf_pos) @ w
    # plot_approximation(estimation, training_target_sin)
    plot_scatter_stuff(x_training, estimation, testing_target_sin)


def main():
    # Do the competitive learning
    do_competitive_learning_()


if __name__ == "__main__":
    main()

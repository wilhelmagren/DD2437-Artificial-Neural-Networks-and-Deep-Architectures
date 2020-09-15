import numpy as np
import matplotlib.pyplot as plt
# Global variables don't touch please
N = 63  # Number of inputs
n = 20 # Number of RBF's, has to be greater than N
step_size = 0.1  # Used for generating sin wave
sigma = 1  # Variance for all nodes


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


def phi_func(x, my):
    return np.exp(-(x - my)**2 / (2*(sigma ** 2)))


def calculate_error(target,estimate):
    return 1/2*((target-estimate)**2)


def delta_learning_rule(error, phi, k, eta):
    return eta*error*phi[k]

def generate_big_phi(input_matrix, rbf_pos):
    big_phi_matrix = np.zeros([N, n])
    for i in range(N):
        for j in range(n):
            big_phi_matrix[i][j] = phi_func(input_matrix[i], rbf_pos[j])
    return big_phi_matrix

def generate_weight():
    return np.random.uniform(0, 2*np.pi, n)


def delta_rule(square):
    train_x, test_x, sin_train_t, sin_test_t, square_train_t, square_test_t = generate_input()
    w =  generate_weight()
    rbf_pos = competetive_rbf(train_x,n,0.1,20000)
    phi_test = generate_big_phi(test_x, rbf_pos)
    phi_train = generate_big_phi(train_x, rbf_pos)
    epochs = 1000
    train_size = len(train_x)
    error = 0
    estimation = []
    for i in range(epochs):
        # print(f" Epoch number: [{i}]")
        if not square:
            for k in range(train_size):
                # print(f" k number: [{k}]")
                e = sin_train_t[k] - phi_train[k]@w
                # k = (k+1) % train_size
                delta_w = delta_learning_rule(e, phi_test, k, 0.01)
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
        error = np.abs(estimation-sin_test_t).mean()
    else:
        estimation = phi_test @ w
        for i in range(len(estimation)):
            if estimation[i] >= 0:
                estimation[i] = 1
            else:
                estimation[i] = -1
        error = np.abs(estimation-square_test_t).mean()
    return error, sin_test_t, estimation

def generate_initial_rbf_position():
    """
    Func generate_initial_rbf_position/0
    @spec generate_initial_rbf_position() :: np.array()
        Generate a uniformed list of starting positions for the RBF's

        TODO: Find out how to most accurately position the RBF's to predict the target function
    """
    rbf_pos = np.random.uniform(0, 2*np.pi, n)
    return rbf_pos

def competetive_rbf(x_training,nodes,eta,epochs):
    RBF = x_training[:n]
    for i in range(epochs):
        random_data_point = x_training[np.random.randint(0,len(x_training))-1]
        dist = np.zeros(nodes)
        for i in range(len(RBF)):
            dist[i] = np.linalg.norm(RBF[i]-random_data_point)
        RBF[np.argmin(dist)] += eta* (random_data_point-RBF[np.argmin(dist)])
    return RBF

def plot_approximation(estimate, target):
    """
    Func plot_approximation/2
    @spec plot_approximation(list, list) :: void
        Plot the function approximation estimate over the target function.
    """
    #print(estimate)
    plt.plot(estimate, label="Estimate", color="red")
    plt.plot(target, label="Target", color="blue")
    plt.title("Estimate vs target")
    plt.grid()
    plt.legend()
    plt.show()


def main():
    global n
    #train_x, test_x, sin_train_t, sin_test_t, square_train_t, square_test_t = generate_input()
    #rbf_pos = competetive_rbf(train_x,n,0.01,1000)



    error,target,est = delta_rule(False)
    #print(error)
    plot_approximation(est, target)
    error_list = []
    """for i in range(50):
        print(f"n is: [{n}] and sigmaballs is: [{sigma}]")
        error, target, est = delta_rule(False)
        error_list.append(error)
        print(f"    error is: [{error}]")
        # plot_approximation(est, target)
        n += 1
    plot_error(error_list, [i for i in range(50)])"""
    #est, tar, error_lists, it_list, rbf_positions = perform_least_squared(True)


if __name__ == "__main__":
    main()

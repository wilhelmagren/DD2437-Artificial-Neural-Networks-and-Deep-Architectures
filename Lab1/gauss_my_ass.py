import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

"""
    BISMILLAH DON'T CURSE OUR CODE 
"""
hidden_neurons = 20
"""
    global variables above don't move
"""


def phi(x):
    return (2 / (1 + np.exp(-x))) - 1


def phi_prime(x):
    return ((1 + phi(x)) * (1 - phi(x))) / 2


def forward_pass(X, W, V):
    hin = W @ X
    hout = np.concatenate((phi(hin), np.ones((1, X.shape[1]))))
    oin = V @ hout
    out = phi(oin)
    return out, hout


def back_pass(out, hout, T, V):
    delta_o = (out-T) * phi_prime(out)
    delta_h = (np.transpose(V) @ delta_o) * phi_prime(hout)
    # Gotta remove the last part(bias)
    delta_h = delta_h[:hidden_neurons, :]
    return delta_h, delta_o


def get_delta_weights(delta_h, delta_o, X, eta, h_out):
    delta_W = -eta * delta_h@np.transpose(X)
    delta_V = -eta * delta_o @ np.transpose(h_out)
    return delta_W, delta_V


def update_weights(V, W, delta_W, delta_V):
    V = V + delta_V
    W = W + delta_W
    return V,W


def calculate_error(X, W, T):
    return np.square(np.subtract(T, W@X)).mean()


def generate_gauss_distribution():
    """
    Func generate_gauss_distribution/0
    @spec generate_gauss_distribution() :: np.array(), np.array(), np.array(), np.array()
    """
    x = np.arange(-5, 5, 0.5)
    y = np.arange(-5, 5, 0.5)
    x_mesh, y_mesh = np.meshgrid(x, y, sparse=False)
    xx, yy = x_mesh.flatten(), y_mesh.flatten()
    X = np.vstack([xx, yy, np.ones(len(xx))])
    T = np.atleast_2d(np.exp(-xx * xx * 0.1) * np.exp(-yy * yy * 0.1) - 0.5)
    W = np.random.normal(1, 0.05, (hidden_neurons, X.shape[0]))
    V = np.random.normal(1, 0.05, (1, hidden_neurons + 1))
    return X, W, V, T


def two_layer_train(X, W, V, T, epoch=1000, eta=0.001):
    """
    Func two_layer_train/6
    @spec two_layer_train(np.array(), np.array(), np.array(), np.array(), integer, integer) :: np.array(), np.array(), list, list
    """
    mse_list = []
    iteration_list = []
    iteration_num = 0
    estimated_out = 0
    for i in range(epoch):
        if (i % 100) == 0:
            print(mse_list)

        o_out, h_out = forward_pass(X, W, V)

        # Doing this only to plot approximated function after training
        estimated_out = o_out

        delta_h, delta_o = back_pass(o_out, h_out, T, V)
        delta_w, delta_v = get_delta_weights(delta_h, delta_o, X, eta, h_out)
        V, W = update_weights(V, W, delta_w, delta_v)
        mse_list.append(calculate_error(X, W, T))
        iteration_list.append(iteration_num)
        iteration_num += 1
    return W, V, mse_list, iteration_list, estimated_out


def plot_function(z):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    x_plot = np.arange(-5, 5, 0.5)
    y_plot = np.arange(-5, 5, 0.5)
    x_plot, y_plot = np.meshgrid(x_plot, y_plot)

    # Reshape the output to grid form of x_plot & y_plot
    estimated_val = np.reshape(z, [20, 20])
    surface = ax.plot_surface(x_plot, y_plot, estimated_val, cmap=cm.coolwarm, antialiased=True)
    plt.show()


def perform_function_approx():
    x, w, v, t = generate_gauss_distribution()
    updated_w, updated_v, mse_list, iteration_list, estimated_out = two_layer_train(x, w, v, t, 10000, 0.001)
    plot_function(estimated_out)


def main():
    perform_function_approx()


if __name__ == "__main__":
    print("### -- Doing main -- ###")
    main()

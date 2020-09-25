import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

"""
    DO NOT TOUCH THESE GLOBAL VARIABLES!
    IF YOU DO HOWEVER TOUCH THEM, THE CODE WILL BE CURSED AND OUR MISSIONS IS DOOMED TO FAIL.
    SAURON WILL REGAIN HIS FULL STRENGTH AND BRING DARKNESS UPON MIDDLE EARTH. 
    THE WORLD OF MEN AS WE NOW IT, WILL COME TO AN END... the age of the Orcs will begin.
"""
# P is the number of patterns, we start with 3?
P = 300
# N is the number of units, what the fuck does that mean. Maybe number of neurons? Haven't implemented any neurons doe..
N = 100
# T_P is the number of patterns we train on
T_P = 300
# PERCENTAGE TO FLIP
flipper = 20
# AVERAGE
ro = 0.01


def energy_function(w, pattern):
    """
    Lyapunov function YAAAAAAAAAAS QUEEN caslculate the energy from a pattern

    # ENERGY IS HIGHER FOR ATTRACTORS
    energy = 0
    for i in range(N):
        inner_sum = 0
        for j in range(N):
            inner_sum += w[i][j] * pattern[i] * pattern[j]
        energy += inner_sum"""

    return - np.einsum('ij,i,j', w, pattern, pattern)


def generate_image(pattern):
    """
    GENERATE THE IMAGE FROM A PATTERN
    """
    pattern = pattern.reshape((10, 10))
    plt.imshow(pattern, interpolation='nearest')
    plt.show()


def plot_acc(acc, it, lbl):
    plt.plot(it, acc, label=lbl)
    plt.grid()
    plt.legend()
    plt.ylabel(f"Number of stable points")
    plt.xlabel("Number of patterns learned")


def sign(x):
    return np.where(x < 0, -1, 1)


def update_rule(pattern, weights, bias):
    """
    @spec hopfield_recall(np.array(), np.array()) :: np.array()
        Recalling a pattern of activation x. The implemented variant is the 'Little model'.
    """
    return 0.5 + 0.5*sign(np.dot(pattern, weights) - bias)


def flip_pattern(patterns):
    for i in range(len(patterns)):
        idx = np.random.choice(N, flipper, replace=False)
        patterns[i][idx] *= -1
    return patterns


def accuracy(target, pattern):
    """
    ☂☂☂☂☂☂☂☂☂☂☂☂☂☂☂☂
    """
    c = 0
    for i, p in enumerate(pattern):
        if target[i] == p:
            c += 1
    return c/N


def num_stable_states(w, pattern):
    patterns = pattern.copy()
    count = 0
    for p in range(len(patterns)):
        for i in range(N):
            sum = 0
            for j in range(N):
                sum += w[i][j] * patterns[p][j]
            if sum >= 0:
                sum = 1
            else:
                sum = -1
            patterns[p][i] = sum
        if np.array_equal(patterns[p], pattern[p]):
            count += 1
    return count


def num_stable_states_batch(w, pattern, pure):
    patterns = pattern.copy()
    count = 0
    for i, p in enumerate(patterns):
        recalled = update_rule(w, p)
        if np.array_equal(recalled, pure[i]):
            count += 1
    return count


def little_model(w, pattern, diagonal=False):
    for p in pattern:
        w += np.outer(p, p)
    if diagonal:
        np.fill_diagonal(w, 0)
    return w


def train(pattern, pure, noise=True):
    num_stable = []
    w = np.zeros((N, N))
    w += np.outer(pure[0], pure[0])
    for i in range(1, 100):
        print(f"Number of patterns trained: {i}")
        w += np.outer(pure[i], pure[i])
        # np.fill_diagonal(w, 0)
        # Calculate number of stable patterns
        # num_stable.append(num_stable_states_batch(w, pattern[:i + 1], pure[:i + 1]))
        count = 0
        for k in range(i - 1):
            correct_pattern = pure[k]
            recalled = update_rule(w, pattern[k])
            if np.array_equal(recalled, correct_pattern):
                count += 1
        num_stable.append(count/i)
    lbl = "clean"
    if noise:
        lbl = "noisy"
    plot_acc(num_stable, [i for i in range(len(num_stable))], lbl)

"""
def sparse_update_rule(w, pattern):
    return 0.5 + 0.5*np.sign(np.dot(w, pattern) - theta)
"""


def sparse_learning_rule(w, pattern, avg_p, ppp):
    for i in range(N):
        for j in range(N):
            w[i][j] = (pattern[i] - avg_p)*(pattern[j] - avg_p)
    return w


def average_activity(pattern):
    sum = 0
    for i in range(P):
        for j in range(N):
            sum += pattern[i][j]
    sum /= (N*P)
    return sum


def create_random_pattern(Nb_of_patterns, length, threshold):
    pattern = np.random.rand(Nb_of_patterns, length)
    pattern = np.where(pattern < threshold, 1, 0)
    return pattern


def calc_weights(pattern, normalize=False, remove_selfcons=False):
    rho = 1/(pattern.shape[0]*pattern.shape[1])*np.sum(pattern)
    pattern = pattern - rho
    weights = pattern.T.dot(pattern)
    if normalize:
        weights = weights/pattern.shape[1]
    if remove_selfcons:
        np.fill_diagonal(weights, 0)
    return weights


def is_pattern_stable(pattern, weights, bias):
    return (pattern == update_rule(pattern, weights, bias)).all()


def main():
    print("in main")
    pattern = create_random_pattern(P, N, ro)
    w = calc_weights(pattern)
    np.max(w)
    theta_map = np.arange(1, 10, 3)
    for theta in theta_map:
        print(f"theta = {theta}")
        num_stable = []
        for i in range(P):
            count = 0
            print(f"Number of patterns trained: {i}")
            w = calc_weights(pattern[:i + 1], remove_selfcons=True)
            for k in range(i + 1):
                if is_pattern_stable(pattern[k], w, theta):
                    count += 1
            num_stable.append(count / (i + 1))
        plot_acc(num_stable, [i for i in range(len(num_stable))], f"theta = {theta}")
    plt.show()


if __name__ == "__main__":
    print("\n### -- hopfield.py Lab3 Part 3.1 -- ###\n")
    main()

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
flipper = 10


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
    plt.ylabel("Number of stable points")
    plt.xlabel("Number of patterns learned")


def update_rule(w, patterns):
    """
    @spec hopfield_recall(np.array(), np.array()) :: np.array()
        Recalling a pattern of activation x. The implemented variant is the 'Little model'.
    """
    return np.sign(np.dot(patterns, w))


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
    for i in range(1, P):
        print(f"Number of patterns trained: {i}")
        w += np.outer(pure[i], pure[i])
        np.fill_diagonal(w, 0)
        # Calculate number of stable patterns
        # num_stable.append(num_stable_states_batch(w, pattern[:i + 1], pure[:i + 1]))
        count = 0
        for k in range(i - 1):
            correct_pattern = pure[k]
            recalled = update_rule(w, pattern[k])
            if np.array_equal(recalled, correct_pattern):
                count += 1
        num_stable.append(count/(i))
    lbl = "clean"
    if noise:
        lbl = "noisy"
    plot_acc(num_stable, [i for i in range(len(num_stable))], lbl)


def main():
    print("in main")
    random_patterns = np.random.randint(-1, 1, (P, N))
    for i in range(len(random_patterns)):
        for j in range(len(random_patterns[0])):
            if random_patterns[i][j] >= 0:
                random_patterns[i][j] = 1
            else:
                random_patterns[i][j] = -1
    noise_patterns = flip_pattern(random_patterns.copy())
    train(random_patterns.copy(), random_patterns.copy(), False)
    train(noise_patterns.copy(), random_patterns.copy())
    plt.show()


if __name__ == "__main__":
    print("\n### -- hopfield.py Lab3 Part 3.1 -- ###\n")
    main()

import numpy as np
import matplotlib.pyplot as plt

"""
    DO NOT TOUCH THESE GLOBAL VARIABLES!
    IF YOU DO HOWEVER TOUCH THEM, THE CODE WILL BE CURSED AND OUR MISSIONS IS DOOMED TO FAIL.
    SAURON WILL REGAIN HIS FULL STRENGTH AND BRING DARKNESS UPON MIDDLE EARTH. 
    THE WORLD OF MEN AS WE NOW IT, WILL COME TO AN END... the age of the Orcs will begin.
"""
# P is the number of patterns, we start with 3?
P = 3
# N is the number of units, what the fuck does that mean. Maybe number of neurons? Haven't implemented any neurons doe..
N = 8


def hopfield_recall(w, patterns):
    """
    @spec hopfield_recall(np.array(), np.array()) :: np.array()
        Recalling a pattern of activation x. The implemented variant is the 'Little model'.
    """
    res = np.dot(patterns, w)
    res = np.sign(res)
    return res


def calculate_weight_matrix(w, pattern_list, scale=False):
    """
    @spec calculate_weight_matrix(np.array(), np.array()) :: np.array()
        Returns a modified weight matrix, where all weights have been calculated given formula below.
        Function currently scales the weight with 1/N. Maybe not?
        TODO: implement feature to choose scaling or not.
    """
    for i in range(N):
        for j in range(N):
            for mu in range(P):
                w[i][j] += pattern_list[mu][i]*pattern_list[mu][j]
            if scale:
                w[i][j] /= N
    return w


def generate_weight_matrix():
    w = np.random.normal(0, 1, (N, N))
    return w


def generate_input_patterns():
    # Since this is distorted with a 1-bit error - maybe the pattern should be: [1, -1, 1, -1, 1, -1, 1, -1] ?
    x1d = np.array([1, -1, 1, -1, 1, -1, -1, 1])
    # 2-bit error
    x2d = np.array([1, 1, -1, -1, -1, 1, -1, -1])
    # 2-bit error
    x3d = np.array([1, 1, 1, -1, 1, 1, -1, 1])
    return np.vstack([x1d, x2d, x3d])


def main():
    """
    Hebbian learning: (non-supervised)
        - Correlated activity causes respective synapses to grow.
            This strengthens connections between correlated neurons.
        - Hebbian learning NN are noise resistant because of anti-correlation.
        - You get easier calculations when using bipolar units [-1, 1] instead of binary units [0, 1].
        - The desired activations of the training patterns are vectors x^u with components x^u _i.
            u is the number of the pattern.
        - Measuring correlated activities with outer product (W=x^t @ x) of the activity vectors we intend to learn.
            Correlation between x_i & x_j -> w_i_j > 0, otherwise anti-correlation w_i_j < 0.
        - W i symmetric, each pair of units will be connected to each other with the same strength.
            TODO: does this mean w_i_j = w_j_i ?
        - The coefficients of W can be calculated as:
            w_i_j = 1/N * sum[from u=1 to P] (x^u_i * x^u_j)
            u is the index within the set of patterns
            P is the number of patterns
            N is the number of units.

    Hopfield network recall:
        - To recall a pattern of activations x in this network we use the update rule in the following way:
            x_i <- sign(sum[all j] (w_i_j * x_j)
        - Synchronously updated is called the 'Little model'. This affects convergence property.
        - When calculating W, we only need to scale with 1/N if using bias, calculating energy or
            approximating complex functions.
        - Fixed points are points that when applying the update rule -> you get the same points back.
    """
    pattern_l = generate_input_patterns()
    w = calculate_weight_matrix(generate_weight_matrix(), pattern_l)
    recalled = hopfield_recall(w, pattern_l[0])
    # If below is true - then the network as been trained to recall a 1-bit error pattern. That's great! :)
    # recalled == pattern_l[0]


if __name__ == "__main__":
    print("\n### -- hopfield.py Lab3 Part 3.1 -- ###\n")
    main()

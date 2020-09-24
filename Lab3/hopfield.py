import numpy as np
import matplotlib.pyplot as plt

"""
    DO NOT TOUCH THESE GLOBAL VARIABLES!
    IF YOU DO HOWEVER TOUCH THEM, THE CODE WILL BE CURSED AND OUR MISSIONS IS DOOMED TO FAIL.
    SAURON WILL REGAIN HIS FULL STRENGTH AND BRING DARKNESS UPON MIDDLE EARTH. 
    THE WORLD OF MEN AS WE NOW IT, WILL COME TO AN END... the age of the Orcs will begin.
"""
# P is the number of patterns, we start with 3?
P = 11
# N is the number of units, what the fuck does that mean. Maybe number of neurons? Haven't implemented any neurons doe..
N = 1024
# T_P is the number of patterns we train on
T_P = 3


def accuracy(target, pattern):
    """
    ☂☂☂☂☂☂☂☂☂☂☂☂☂☂☂☂
    """
    c = 0
    for i, p in enumerate(pattern):
        if target[i] == p:
            c += 1
    return c/N


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


def read_file(path=".\\pict.dat"):
    """
    READ THE FILE FROM THE PATH and return an array of the data
    """
    f = open(path, "r")
    data = f.readline()
    data = data.split(",")
    parsed_data = [[] for i in range(P)]
    for i in range(P):
        for d in range(N):
            parsed_data[i].append(int(data.pop(0)))
    f.close()
    # print(parsed_data)
    return np.array(parsed_data)


def generate_image(pattern):
    """
    GENERATE THE IMAGE FROM A PATTERN
    """
    pattern = pattern.reshape((32, 32))
    plt.imshow(pattern, interpolation='nearest')
    plt.show()


def hopfield_recall(w, patterns):
    """
    @spec hopfield_recall(np.array(), np.array()) :: np.array()
        Recalling a pattern of activation x. The implemented variant is the 'Little model'.
    """
    res = np.dot(patterns, w)
    res = np.sign(res)
    return res


def test_sequential_hopfield(w, dist_p):
    """
    @spec test_sequential_hopfield(np.array(), np.array()) :: void
        Sequentially and randomly chose pattern and try to recall it
    """
    energy_list = []
    for p in range(len(dist_p)):
        rand_p = p
        count = 0
        for _ in range(N):
            i = np.random.randint(0, N)
            sum = 0
            for j in range(N):
                sum += w[i][j] * dist_p[rand_p][j]
            if sum >= 0:
                sum = 1
            else:
                sum = -1
            dist_p[rand_p][i] = sum
            #if count % 100 == 0:
            energy_list.append(energy_function(w, dist_p[p]))
            #generate_image(dist_p[rand_p])
            count += 1
        return energy_list


def test_hopfield(w, dist_p):
    """
    @spec test_hopfield(np.array(), np.array()) :: void
        Iteratively try and recall the original pattern given a distorted pattern.
        This recalled pattern is called a 'Fixed point' if we apply the update_rule and get the same pattern back.
    """
    # Iteratively call hopfield_recall/2 here to see convergence until recalled same pure pattern.
    for p in dist_p:
        tmp_p = p
        recalled_pattern = hopfield_recall(w, p)
        while not np.array_equal(recalled_pattern, p):
            # print(f"The pattern {p} did not reach fixed point within {count} iterations.\n    |-> The last recalled"
            #      f" pattern looks like {recalled_pattern}")
            p = recalled_pattern
            recalled_pattern = hopfield_recall(w, p)
        print(f"The pattern vector {tmp_p} and recalled fixed point {recalled_pattern}")
        generate_image(tmp_p)
        generate_image(recalled_pattern)
    return


def calculate_weight_matrix(w, pattern_list, scale=True):
    """
    @spec calculate_weight_matrix(np.array(), np.array()) :: np.array()
        Returns a modified weight matrix, where all weights have been calculated given formula below.
        Function currently scales the weight with 1/N. Maybe not?
        TODO: implement feature to choose scaling or not.
    """

    for i in range(N):
        for j in range(N):
            for mu in range(T_P):
                w[i][j] += pattern_list[mu][i]*pattern_list[mu][j]
            if scale:
                w[i][j] /= N
    return w


def generate_weight_matrix():
    """
    @spec generate_weight_matrix() :: np.array()
        Return a normally distributed NxN weight matrix.
        TODO: Look at other ways to initialize the weight matrix. Also maybe we should use bias?
    """
    w = np.random.normal(0,1,(N, N))

    return w


def generate_input_patterns():
    """
    @spec generate_input_patterns() :: np.array(), np.array()
        Create the pure training patterns and the distorted testing patterns.
        Return them as PxN np.array() with pure matrix being returned as first matrix.
    """
    # x1, x2, x3 are the initial patterns which we will train the network on.
    # x1 should be 1 bit different from x1d
    x1 = np.array([1, -1, 1, -1, 1, -1, 1, 1])
    # x2 should be 2-bit different from x2d
    x2 = np.array([1, 1, -1, -1, 1, 1, -1, -1])
    # x3 should also be 2-bit different from x3d
    x3 = np.array([1, -1, 1, -1, 1, 1, -1, -1])

    # Since this is distorted with a 1-bit error - maybe the pattern should be: [1, -1, 1, -1, 1, -1, 1, -1] ?
    x1d = np.array([1, -1, 1, -1, -1, 1, -1, -1])
    # 2-bit error
    x2d = np.array([1, 1, -1, -1, -1, -1, 1, 1])
    # 2-bit error
    x3d = np.array([1, -1, 1, -1, -1, -1, 1, 1])
    return np.vstack([x1, x2, x3]), np.vstack([x1d, x2d, x3d])


def plot_acc(acc, it):
    plt.plot(it, acc, color="green")
    plt.grid()
    plt.ylabel("Energy")
    plt.xlabel("number of iterations")
    plt.show()


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

    Maybe look at this for some theory: https://www.youtube.com/watch?v=nv6oFDp6rNQ
    """
    data = read_file()
    # pure_patterns, dist_patterns = generate_input_patterns()
    mod_data = data[:3, :N]
    w = calculate_weight_matrix(generate_weight_matrix(), mod_data)
    w = np.random.normal(0,1,(N,N))
    wsym = 0.5 * (w*w.T)
    # for p in mod_data:
    #    print(energy_function(w, p))
    # print(energy_function(w, data[5]))
    # W is now set up from the pure training patterns. We will now see how well the network can recall the
    #       training patterns based on distorted versions of them. Remember to look at convergence rate!
    acc_l = test_sequential_hopfield(w, [data[9], data[10]])
    acc_l1 = test_sequential_hopfield(wsym, [data[9], data[10]])
    #print(acc_l)
    #print(acc_l[1])
    it_l = [i for i in range(len(acc_l))]
    plot_acc(acc_l, it_l)
    plot_acc(acc_l1, it_l)

    # recalled_pattern = hopfield_recall(w, pattern_l[1])
    # print(recalled_pattern)
    # If below is true - then the network as been trained to recall a 1-bit error pattern. That's great! :)
    # recalled_pattern == pattern_l[0]


if __name__ == "__main__":
    print("\n### -- hopfield.py Lab3 Part 3.1 -- ###\n")
    main()

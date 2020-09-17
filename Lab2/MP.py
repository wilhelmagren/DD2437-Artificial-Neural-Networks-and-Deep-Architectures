
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import re

members = 349
votes = 31
n = 100


def read_data_votes(voting):
    f = open(voting,"r")
    animals = np.zeros((members,votes))
    line = f.read().split(",")
    for i in range(members):
        for j in range(votes):
            animals[i][j] = float(line.pop(0))
    f.close()
    return animals


def read_party(path):
    f = open(path, "r")
    parties = f.read()
    parties = re.sub(r'\t', '', parties)
    parties = re.sub(r' ', '', parties)
    parties = list(map(int, parties.splitlines()))
    return np.asarray(parties)


def find_closest(animal, W, num_neighbour, grannar=True, eta=0.2):
    """
    Func find_closest/5
    @spec find_closest(np.array(), np.array(), integer, boolean, integer) :: np.array()
        Given a data-sample 'animal', a weight matrix 'W', an integer of how many neighbours should be used
        -> update the weight matrix W based on the number of neighbours.
    """
    dist = np.zeros(n)
    for i, w in enumerate(W):
        dist[i] = np.linalg.norm(animal-w)

    if not grannar:
        return np.argmin(dist)

    for index in range(num_neighbour):
        tmp_index = 0
        if np.argmin(dist) + index > len(W) - 1:
            tmp_index = np.argmin(dist) + index - len(W)
            W[tmp_index] += eta * (animal - W[tmp_index])
            if np.argmin(dist) - index < 0:
                tmp_index = np.argmin(dist) - index + len(W) - 1
                W[tmp_index] += eta * (animal - W[tmp_index])
            else:
                W[np.argmin(dist) - index] += eta * (animal - W[np.argmin(dist) - index])
        else:
            W[np.argmin(dist) + index] += eta * (animal - W[np.argmin(dist) + index])
            if np.argmin(dist) - index < 0:
                tmp_index = np.argmin(dist) - index + len(W) - 1
                W[tmp_index] += eta * (animal - W[tmp_index])
            else:
                W[np.argmin(dist) - index] += eta * (animal - W[np.argmin(dist) - index])
    return W


def save_the_animals(input, epoch=20):
    """
    Func save_the_animals/2
    @spec save_the_animals(list, integer) :: void
        Find the correct solution to the given minimization problem.
        Currently finds a short (maybe shortest) path for the Travel Sales Person problem
        which is known to be unsolvable in polynomial time complexity.
    """
    w = np.random.rand(votes, n)
    w = w.T
    # Update W
    for e in range(epoch):
        for vote in input:
            # This neighboy variable is the solution to everything - don't ever question why we use this formula.
            # The formula to calculate the number of neighbours came to me in a dream - and it is our highest truth.
            neighboy = round((n - e)/2)
            w = find_closest(vote, w, neighboy)
    # Route array used for storing argmin of weight vector rows
    pos = np.zeros(n)
    for i in range(n):
        # Just something big
        distance = np.zeros(n)
        # Pick out the data point
        data_point_x = input[i, :]
        for j in range(n):
            # Calculate the the distance from the point x to all weights of row j in W
            distance[j] = euclidean(data_point_x, w[j, :])
        pos[i] = np.argmin(distance)
    # Now we have the route
    ordered = np.argsort(pos)
    ordered = np.concatenate((ordered, ordered))
    return ordered


def map_pos_to_attributes(res, parties):
    """
        % Coding: 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'
        % Use some color scheme for these different groups
    """
    num_col_part = {
        0: "grey",
        1: "blue",
        2: "yellow",
        3: "red",
        4: "orange",
        5: "green",
        6: "magenta",
        7: "cyan"
    }
    num_col_sex = {
        0: "red",
        1: "blue",
    }

    # pos_party = np.concatenate((res, parties))
    for v in range(len(res)):
        col = num_col_part[int(parties[v])]
        plt.scatter(res[v], parties[v], color=col)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.xlabel("left - right")
    plt.ylabel("lib - authoritarian")
    plt.title("Party votes")
    plt.show()


def main():
    print("### -- In main cyclic.py -- ###")
    votes_input = read_data_votes(".\\datasets\\votes.dat")
    parties = read_party(".\\datasets\\mpparty.dat")
    res = save_the_animals(votes_input)
    map_pos_to_attributes(res, parties)


if __name__ == "__main__":
    main()

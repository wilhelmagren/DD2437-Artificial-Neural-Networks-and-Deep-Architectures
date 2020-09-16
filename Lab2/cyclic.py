import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean


"""
    GLOBAL VARIABLES DO NOT TOUCH!
        the code will be cursed if you touch anything - maybe don't even look at the code...
"""
species = 32
attributes = 2
n = 10
sigma = 1


def read_data_cities(cities):
    """
    Func read_data_cities/1
    @spec read_data_cities(string) :: np.array()
        Takes the file path and reads the file - returns the parsed file as an input array for our Neural Network
    """
    f = open(cities, "r")
    city_map = []
    for line in f:
        city_map.append(line.split("\n")[0].split(","))
    training = np.asfarray(np.array(city_map), float)
    return training


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
    w = np.random.rand(2, n)
    w = w.T
    # Update W
    for e in range(epoch):
        for animal in input:
            # This neighboy variable is the solution to everything - don't ever question why we use this formula.
            # The formula to calculate the number of neighbours came to me in a dream - and it is our highest truth.
            neighboy = round((n - e)/2)
            w = find_closest(animal, w, neighboy)
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
    # Add the starting position as last position in the route - since we are doing a cycle
    ordered = np.append(ordered, ordered[0])
    print(f"The ordered route looks like this:\n    |-> {ordered}")
    plt.title("Cyclic route")
    plt.scatter(input[ordered][:, 0], input[ordered][:, 1])
    plt.plot(input[ordered][:, 0], input[ordered][:, 1])
    plt.show()


def main():
    print("### -- In main cyclic.py -- ###")
    city_input = read_data_cities(".\\datasets\\cities.dat")
    save_the_animals(city_input)


if __name__ == "__main__":
    main()

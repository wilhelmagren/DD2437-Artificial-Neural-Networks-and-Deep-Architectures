import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

species = 32
attributes = 2
n = 10
sigma = 1


def read_data_cities(cities):
    f = open(cities, "r")
    city_map = []
    for line in f:
        city_map.append(line.split("\n")[0].split(","))
    training = np.asfarray(np.array(city_map), float)
    return training


def find_closest(animal, W, num_neighbour, grannar=True, eta=0.2):
    """
    RETURNS WEIGHT MATRIX UPDATED
    """
    dist = np.zeros(n)
    for i, w in enumerate(W):
        dist[i] = np.linalg.norm(animal-w)

    if not grannar:
        return np.argmin(dist)
    """
    LEGACY CODE
    
    for index in range(num_neighbour):
        if 0 < np.argmin(dist) + index < attributes:
            W[np.argmin(dist) + index] += eta * (animal - W[np.argmin(dist) + index])
        if 0 < np.argmin(dist) - index < attributes:
            W[np.argmin(dist) - index] += eta * (animal-W[np.argmin(dist) - index])"""
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
    """if e <= epoch/2:
        num_neighbour = 1
    if epoch - e < 5:
        num_neighbour = 0
    for index in range(num_neighbour):
        if np.argmin(dist) == 0:
            W[9] += eta * (animal - W[9])
            W[1] += eta * (animal - W[1])
        elif np.argmin(dist) == 9:
            W[8] += eta * (animal - W[8])
            W[0] += eta * (animal - W[0])
        else:
            W[np.argmin(dist) + index] += eta * (animal - W[np.argmin(dist) + index])
            W[np.argmin(dist) - index] += eta * (animal - W[np.argmin(dist) - index])"""
    return W


def save_the_animals(input, epoch=20):
    w = np.random.rand(2, n)
    w = w.T
    # Update W
    for e in range(epoch):
        for animal in input:
            neighboy = round((n - e)/2)
            w = find_closest(animal, w, neighboy)
    return w


def find_closest_w_row(x, w):
    d = np.zeros(10)
    for i in range(10):
        d[i] = euclidean(x, w[i, :])
    return np.argmin(d)


def order_cities(cities, w):
    pos = np.zeros(10)
    for i in range(10):
        ind = find_closest_w_row(cities[i, :], w)
        pos[i] = ind
    order = np.argsort(pos)
    order = np.append(order, order[0])
    print(order)
    plt.title("Cyclic tour")
    plt.scatter(cities[order][:, 0], cities[order][:, 1])
    plt.plot(cities[order][:, 0], cities[order][:, 1])
    plt.show()


def main():
    print("### -- In main part2_som.py -- ###")
    city_input = read_data_cities(".\\datasets\\cities.dat")
    w = save_the_animals(city_input)
    order_cities(city_input, w)


if __name__ == "__main__":
    main()

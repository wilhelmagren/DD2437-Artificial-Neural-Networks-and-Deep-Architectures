import numpy as np
import matplotlib.pyplot as plth

species = 32
attributes = 2
n = 10
sigma = 1

def read_data_cities(cities):
    f = open(cities,"r")
    city_map = []
    for line in f:
        city_map.append(line.split("\n")[0].split(","))



def find_closest(animal, W, epoch, grannar=True, eta=0.2):
    """
    RETURNS WEIGHT MATRIX UPDATED
    """
    dist = np.zeros(n)
    for i, w in enumerate(W):
        dist[i] = np.linalg.norm(animal-w)

    if not grannar:
        return np.argmin(dist)

    num_neighbour = 50 - epoch*4
    if num_neighbour < 0:
        num_neighbour = 1
    for index in range(num_neighbour):
        if 0 < np.argmin(dist) + index < attributes:
            W[np.argmin(dist) + index] += eta * (animal - W[np.argmin(dist) + index])
        if 0 < np.argmin(dist) - index < attributes:
            W[np.argmin(dist) - index] += eta * (animal-W[np.argmin(dist) - index])
    return W


def save_the_animals(input, epoch=50):
    w = np.random.rand(n, attributes)
    # Update W
    for e in range(epoch):
        for animal in input:
            w = find_closest(animal, w, e)

    # Loop through all animals once more
    pos = []
    for animal in input:
        # Calculate the index of the winning node for each animal -> store in vec pos
        argmin = find_closest(animal, w, e, False)
        pos.append(argmin)
    # Sort the vec pos and we get the animals in correct order. EASY?

    return pos


def main():
    print("### -- In main part2_som.py -- ###")
    city_input = read_data_cities(".\\datasets\\cities.dat")


if __name__ == "__main__":
    main()

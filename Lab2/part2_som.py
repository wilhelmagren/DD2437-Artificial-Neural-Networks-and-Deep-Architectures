import numpy as np
import matplotlib.pyplot as plth

species = 32
attributes = 84
n = 100


def read_data_animals(animals):

    f = open(animals,"r")
    animals = np.zeros((species,attributes))
    line = f.read().split(",")
    for i in range(species):
        for j in range(attributes):
            animals[i][j] = int(line.pop(0))
    f.close()
    return animals


def read_data_animals_name(animals_name):
    f = open(animals_name, "r")
    animals_name = []
    for line in f:
        animals_name.append(line.split("\t\n")[0])
    f.close()
    return animals_name


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
    animal = read_data_animals(".\\datasets\\animals.dat")
    animal_names = read_data_animals_name(".\\datasets\\animalnames.txt")
    index = save_the_animals(animal)
    dtype = [('index', int), ('animal', 'S10')]
    values = [0 for i in range(species)]

    for i in range(species):
        values[i] = (index[i], animal_names[i])
    array = np.array(values, dtype=dtype)
    print(np.sort(array, order='index'))


if __name__ == "__main__":
    main()

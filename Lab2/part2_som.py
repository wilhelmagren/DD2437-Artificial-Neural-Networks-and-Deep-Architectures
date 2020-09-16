import numpy as np
import matplotlib.pyplot as plt
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
def find_closest(animal,W,eta=0.2):
    dist = np.zeros(attributes)
    for i,w in enumerate(W):
        dist[i] = np.linalg.norm(animal-w)
    W[np.argmin(dist)] += eta * animal-w

def save_the_animals(input,epoch):
    w = np.random.rand(1,(n,attributes))
    for e in range(epoch):
        for animal in input:
            closest_weight_id = find_closest(animal,w)








def main():
    print("### -- In main part2_som.py -- ###")
    animal = read_data_animals("C:\\Users\\erjab\\PycharmProjects\\pythonProject\\dd2437-ann-new\\dd2437-ann\\Lab2\\datasets\\animals.dat")
    animaL_names = read_data_animals_name("C:\\Users\\erjab\\PycharmProjects\\pythonProject\\dd2437-ann-new\\dd2437-ann\\Lab2\\datasets\\animalnames.txt")
    print(animal)
    print(animaL_names)



if __name__ == "__main__":
    main()

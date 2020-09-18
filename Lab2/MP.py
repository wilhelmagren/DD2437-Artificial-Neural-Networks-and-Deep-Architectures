
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


def read_data_district(voting):
    f = open(voting, "r")
    distr = []
    for line in f:
        distr.append(int(line.split("\n")[0]))
    return distr


def read_data_gender(voting):
    f = open(voting, "r")
    gender = []
    for line in f:
        gender.append(int(line.split("\n")[0]))
    return gender


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
    # num_neighbour = 50 - num_neighbour*3
    if num_neighbour < 0:
        num_neighbour = 1
    for index in range(num_neighbour):
        if 0 < np.argmin(dist) + index < votes:
            W[np.argmin(dist) + index] += eta * (animal - W[np.argmin(dist) + index])
        if 0 < np.argmin(dist) - index < votes:
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
    w = np.random.rand(n, votes)
    # Update W
    for e in range(epoch):
        for vote in input:
            # This neighboy variable is the solution to everything - don't ever question why we use this formula.
            # The formula to calculate the number of neighbours came to me in a dream - and it is our highest truth.
            neighboy = round((n - e)/2)
            w = find_closest(vote, w, neighboy)
    # Loop through all animals once more
    pos = []
    for animal in input:
        # Calculate the index of the winning node for each animal -> store in vec pos
        dist = np.zeros(n)
        # find_closest(animal, w, 0, False)
        for i, we in enumerate(w):
            dist[i] = np.linalg.norm(animal - we)
        pos.append(np.argmin(dist))
    # Make column vector
    """print(pos)
    pos = np.sort(np.array(pos))
    print(pos)
    pos_vec = pos.reshape(-1, 1)
    pos = np.zeros(pos_vec.shape)
    pos = np.concatenate((pos, pos), axis=1)
    for i, e in enumerate(pos):
        tmp_1 = (e // 10) + np.random.normal(0, 0.15)
        tmp_2 = (e % 10) + np.random.normal(0, 0.15)
        tmp_list = [tmp_1[0], tmp_2[0]]
        print(tmp_list)
        tmp_list = np.array(tmp_list)
        pos[i, :] = tmp_list"""
    return pos


def map_pos_to_attributes(res, parties, genders, districts):
    print(f"\nThe training/testing vector looks like this:\n    |-> {res}")
    print(f"\nThe parties data looks like this:\n    |-> {parties}")
    """
        % Coding: 0=no party, 1='m', 2='fp', 3='s', 4='v', 5='mp', 6='kd', 7='c'
        % Use some color scheme for these different groups
        
        Each number in the mpparty.dat file represents what party each corresponding MP is a part of.
    """
    num_col_part = {
        0: "gray",
        1: "blue",
        2: "yellow",
        3: "red",
        4: "orange",
        5: "black",
        6: "magenta",
        7: "green"
    }
    num_col_sex = {
        0: "blue",
        1: "magenta",
    }

    num_col_dist = {
        0: "#FF0000",
        1: "#FF0049",
        2: "#FF008B",
        3: "#FF00F7",
        4: "#E400FF",
        5: "#AA00FF",
        6: "#6C00FF",
        7: "#2E00FF",
        8: "#3E39A0",
        9: "#7D79D8",
        10: "#79D3D8",
        11: "#33E7F2",
        12: "#33F2B5",
        13: "#1F7A3F",
        14: "#164626",
        15: "#111D15",
        16: "#CFCF1C",
        17: "#A0A019",
        18: "#4F4F16",
        19: "#E4B633",
        20: "#DF821F",
        21: "#874C0E",
        22: "#7C0E87",
        23: "#DE95E5",
        24: "#BDBDBD",
        25: "#3C6F8A",
        26: "#BB8ED5",
        27: "#4E2F61",
        28: "#7FEEFF"
    }

    tmp_mod = np.zeros((members, 2))
    for i, e in enumerate(res):
        tmp_1 = ((e // 10) + np.random.normal(0, 0.5))
        tmp_2 = (e % 10) + np.random.normal(0, 0.5)
        tmp_list = [tmp_1, tmp_2]
        tmp_list = np.array(tmp_list)
        tmp_mod[i, :] = tmp_list

    for v in range(members):
        col = num_col_part[int(parties[v])]
        plt.scatter(tmp_mod[v, 0], tmp_mod[v, 1], color=col)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.xlabel("left - right")
    plt.ylabel("lib - authoritarian")
    plt.title("Party votes")
    plt.legend()
    plt.grid(True)
    plt.show()
    for v in range(members):
        col = num_col_sex[int(genders[v])]
        plt.scatter(tmp_mod[v, 0], tmp_mod[v, 1], color=col)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.xlabel("left - right")
    plt.ylabel("lib - authoritarian")
    plt.title("Gender votes")
    plt.legend()
    plt.grid(True)
    plt.show()
    for v in range(members):
        col = num_col_dist[int(districts[v]) - 1]
        plt.scatter(tmp_mod[v, 0], tmp_mod[v, 1], color=col)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.xlabel("left - right")
    plt.ylabel("lib - authoritarian")
    plt.title("District votes")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    print("### -- In main MP.py -- ###")
    votes_input = read_data_votes(".\\datasets\\votes.dat")
    parties = read_party(".\\datasets\\mpparty.dat")
    res_XDXDD = save_the_animals(votes_input)
    values = [0 for i in range(members)]
    genders = read_data_gender(".\\datasets\\mpsex.dat")
    districts = read_data_district(".\\datasets\\mpdistrict.dat")
    dtype = [('index', int), ('party', int), ('gender', int), ('district', int)]
    for i in range(members):
        values[i] = (res_XDXDD[i], parties[i], genders[i], districts[i])
    dbl_array = np.array(values, dtype=dtype)
    res = np.sort(dbl_array, order='index')
    parties = []
    index_new = []
    genders = []
    district = []
    for i in range(len(res)):
        parties.append(res[i]['party'])
        index_new.append(res[i]['index'])
        genders.append(res[i]['gender'])
        district.append(res[i]['district'])
    map_pos_to_attributes(index_new,  parties, genders, districts)


if __name__ == "__main__":
    main()

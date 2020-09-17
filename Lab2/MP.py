
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

members = 349
votes = 31
n = 10
def read_data_votes(voting):
    f = open(voting,"r")
    animals = np.zeros((members,votes))
    line = f.read().split(",")
    for i in range(members):
        for j in range(votes):
            animals[i][j] = float(line.pop(0))
    f.close()
    return animals

def find_closest(animal, W, num_neighbour, grannar=True, eta=0.2):
    """
    Func find_closest/5
    @spec find_closest(np.array(), np.array(), integer, boolean, integer) :: np.array()
        Given a data-sample 'animal', a weight matrix 'W', an integer of how many neighbours should be used
        -> update the weight matrix W based on the number of neighbours.
    """
    dist = np.zeros(n)
    for i, w in enumerate(W):
        dist[i] = np.linalg.norm(animal - w)

    if not grannar:
        return np.argmin(dist)

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
    # Add the starting position as last position in the route - since we are doing a cycle
    ordered = np.append(ordered, ordered[0])
    print(f"The ordered route looks like this:\n    |-> {ordered}")
    plt.title("Cyclic route")
    plt.scatter(input[ordered][:, 0], input[ordered][:, 1])
    plt.plot(input[ordered][:, 0], input[ordered][:, 1])
    plt.show()

def main():
    print("### -- In main cyclic.py -- ###")
    votes_input = read_data_votes(".\\datasets\\votes.dat")
    save_the_animals(votes_input)


if __name__ == "__main__":
    main()

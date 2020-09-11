import numpy as np
import matplotlib.pyplot as plt

"""
 Use Gaussian RBF's to approximate simple functions of one variable.
    Every unit in the hidden layer implement the transfer function phi_i(x)
 The output layer calculates the weighted sum of the n hidden layer units.
 
 (1)   f^(x) = sum from i to n Wi * phi_i(x)
 
 The RBF layer maps the input space to an n-dimensional vector.
 n is usually higher than the dimension of the input space.
 
 We want to find weights which minimize the total approximation error summed 
 over all N patterns used as training examples.
 
 f(xk) => f() is the target function, xk is the k'th pattern, 
 we write a linear equation system with ONE ROW PER PATTERN, where each
 row states the above equation (1) for a particular pattern.
 
 If the number of inputs are greater than the number of units in the hidden layer: N > n
 the system is overdetermined - cannot use Gaussian elimination to solve for W.
 
 Reflect over the questions:
    [] What is the lower bound for the number of training examples, N?
    [] What happens with the error if N = n? Why?
    [] Under what conditions, if any, does the system of linear equations have a solution?
    [] During training we use an error measure defined over the training examples. 
       Is it good to use this measure when evaluating the performance of the network? Explain!

 Two different methods for determining the weights w_i:
    batch modde using least squares
    sequential (instrumental, on-line) learning using the delta rule
"""

def least_squares():
    """
        We can write the system of linear equations as: PHI * w = f

                    ----------------------------------------
                    | phi_1(x1), phi_2(x1), ..., phi_n(x1) |
                    | phi_1(x2), phi_2(x2), ..., phi_n(x2) |
        where PHI = | phi_1(x3), phi_2(x3), ..., phi_n(x3) |
                    | .                                    |
                    | phi_1(xn), phi_2(xn), ..., phi_n(xn) |
                    ----------------------------------------

        and W = [w_1, w_2, ..., w_n]^T

        The function error becomes:
            total error = ||PHI*w - f||^2

        According to linear algebra - obtain W by solving the following system:
            PHI^T * PHI * W = PHI^T * f
    """


def perform_task():
    print("### --- Doing the thing --- ###")


def main():
    perform_task()


if __name__=="__main__":
    main()
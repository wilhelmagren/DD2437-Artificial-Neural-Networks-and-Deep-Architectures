import numpy as np
import matplotlib.pyplot as plt
n = 100
iterations = 1000
hidden_neurons = 3
mA = [1.0,0.3]
mB = [0.0,-0.1]
sigmaA = 0.2
sigmaB = 0.3
step_length = 0

def phi(x):
    return (2 / (1 + np.exp(-x))) - 1
def phi_prime(x):
    return ((1 + phi(x)) * (1 - phi(x))) / 2

def generate_matrices():
    X = np.zeros([3,2*n])
    #Make non linear sets as before, i used normal cuz it was fucked when i did random.rand and concat
    X[0, :n] = np.concatenate((np.random.normal(1, 1, int(n / 2)) * sigmaA - mA[0], np.random.normal(1, 1, int(n / 2)) * sigmaA + mA[0]))
    X[0,n:] = np.random.normal(0,1,n) * sigmaB + mB[0]
    X[1,:n] = np.random.normal(0,1,n) * sigmaA + mA[1]
    X[1,n:] = np.random.normal(0,1,n) * sigmaB + mB[1]
    #added biased
    X[2, :2*n] = 1.0
    # Vikter från X -> Hidden layers
    W = np.random.normal(1,0.5,(hidden_neurons,X.shape[0]))
    #Vikter från Hidden layers -> Output, jag la till 1 så att vi kan hantera bias och matris multiplikation
    V =  np.random.normal(1,0.5,(1,hidden_neurons + 1))
    #T är som vanligt
    T = np.zeros([1,2*n])
    T[0,:n] = 1
    T[0,n:] = -1

    return X,W,V,T

def plot(X):
    plt.scatter(X[0,:n],X[1,:n])
    plt.scatter(X[0,n:],X[1,n:])
    plt.show()
#Forward pass är att gå från X till output
def forward_pass(X,W,V):
    hin = W @ X
    hout = np.concatenate((phi(hin), np.ones((1, X.shape[1]))))
    oin = V @ hout
    out = phi(oin)
    return hout,out

def back_pass(out,hout,T,V):
    delta_o = out-T * phi_prime(out)
    delta_h = (np.transpose(V)@delta_o) * phi_prime(hout)
    #Gotta remove the last part(bias)
    delta_h = delta_h[:hidden_neurons,:]
    return delta_h,delta_o

def get_delta_weights(delta_h,delta_o,X,eta,h_out):
    delta_W = -eta * delta_h@np.transpose(X)
    delta_V = -eta * delta_o @ np.transpose(h_out)
    return delta_W,delta_V

def update_weights(V,W,delta_W,delta_V):
    V = V + delta_V
    W = W + delta_W
    return V,W

def calculate_error(X, W, T):
    return np.sum((T - W@X) ** 2) / 2
#def mean_square_error():
#def get_delta_weights_momentum
def two_layer_train(X,T,W,V,epoch,eta):
    for i in range(epoch):
        """ First we get o_out(final output) and h_out (output from hidden node) with forward pass, 
            then we get the delta_h,delta_o by using back propagation, finally we set deltaW,deltaV and repeat for set epochs"""
        o_out, h_out = forward_pass(X,W,V)
        delta_h, delta_o = back_pass(o_out,h_out,T,V)
        delta_W, delta_V = get_delta_weights(delta_h,delta_o,eta,h_out)
        V,W = update_weights(V,W,delta_W,delta_V)
        return V,W





X,W,V,T = generate_matrices()

plot(X)
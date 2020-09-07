import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
n = 100
iterations = 1000
hidden_neurons = 2
mA = [-2.0, 0.0]
mB = [0.0, 0.0]
sigmaA = 0.2
sigmaB = 0.3
step_length = 0
learning_rate = 0.001


def phi(x):
    return (2 / (1 + np.exp(-x))) - 1

def guass_func():
    x = np.arange(-5,5,0.5)
    y = np.arange(-5,5,0.5)
    xx,yy = np.meshgrid(x,y,sparse=False)
    xx,yy = xx.flatten(),yy.flatten()
    print(xx)
    print(yy)


    #z = np.exp(-x*x*0.1) * np.exp(-y*y*0.1) - 0.5
    #np.mesh(x,y,z)
    #targets = np.reshape(z,1,ndata)
    #[xx,yy] = np.meshgrid(x,y)
    #atterns = [np.reshape(xx,1,ndata):reshape(yy,1,ndata)]

def phi_prime(x):
    return ((1 + phi(x)) * (1 - phi(x))) / 2


def generate_matrices():
    X = np.zeros([3,2*n])
    # Make non linear sets as before, i used normal cuz it was fucked when i did random.rand and concat
    X[0, :n] = np.concatenate((np.random.normal(1, 1, int(n / 2)) * sigmaA - mA[0], np.random.normal(1, 1, int(n / 2)) * sigmaA + mA[0]))
    X[0,n:] = np.random.normal(0,1,n) * sigmaB + mB[0]
    X[1,:n] = np.random.normal(0,1,n) * sigmaA + mA[1]
    X[1,n:] = np.random.normal(0,1,n) * sigmaB + mB[1]
    # added biased
    X[2, :2*n] = 1.0
    # Vikter från X -> Hidden layers
    W = np.random.normal(1, 0.5, (hidden_neurons, X.shape[0]))
    # Vikter från Hidden layers -> Output, jag la till 1 så att vi kan hantera bias och matris multiplikation
    V =  np.random.normal(1,0.5,(1,hidden_neurons + 1))
    # T är som vanligt
    T = np.zeros([1,2*n])
    T[0,:n] = -1
    T[0,n:] = 1

    return X,W,V,T


def plot(X):
    plt.scatter(X[0,:n],X[1,:n])
    plt.scatter(X[0,n:],X[1,n:])
    plt.show()


# Forward pass är att gå från X till output
def forward_pass(X,W,V):
    #print(W.shape)
    hin = W @ X
   # print(hin.shape)
    hout = np.concatenate((phi(hin), np.ones((1, X.shape[1]))))
    oin = V @ hout
    out = phi(oin)
    return out, hout


def back_pass(out,hout,T,V):
    delta_o = (out-T) * phi_prime(out)
    delta_h = (np.transpose(V) @ delta_o) * phi_prime(hout)
    # Gotta remove the last part(bias)
    delta_h = delta_h[:hidden_neurons, :]
    return delta_h, delta_o


def get_delta_weights(delta_h, delta_o, X, eta, h_out):
    delta_W = -eta * delta_h@np.transpose(X)
    delta_V = -eta * delta_o @ np.transpose(h_out)
    return delta_W, delta_V


def update_weights(V, W, delta_W, delta_V):
    V = V + delta_V
    W = W + delta_W
    return V,W


def calculate_error(X, W, T):
    return np.sum((T - W@X) ** 2) / 2


def mse_encoder(X, T, W, V):
    o_out, h_out = forward_pass(X,W,V)
    error = T - np.sign(o_out)
    return np.sum(error ** 2) / len(T),o_out,h_out


# def mean_square_error():
# def get_delta_weights_momentum
def auto_encode(epoch,eta):
    output = 8
    X = np.array([[1,-1,-1,-1,-1,-1,-1,-1],
                 [-1,1,-1,-1,-1,-1,-1,-1],
                 [-1,-1,1,-1,-1,-1,-1,-1],
                 [-1,-1,-1,1,-1,-1,-1,-1],
                 [-1,-1,-1,-1,1,-1,-1,-1],
                 [-1,-1,-1,-1,-1,1,-1,-1],
                 [-1,-1,-1,-1,-1,-1,1,-1],
                 [-1,-1,-1,-1,-1,-1,-1,1],
                 [1,1,1,1,1,1,1,1]])
    W = np.random.normal(1, 0.5, (hidden_neurons, X.shape[0]))
    # Vikter från Hidden layers -> Output, jag la till 1 så att vi kan hantera bias och matris multiplikation
    V = np.random.normal(1, 0.5, (output, hidden_neurons + 1))
    T = X[:-1]
    W,V,acc_list,iterations = two_layer_train(X,T,W,V,epoch,eta)
    #print(np.sign(W))

    error,o_out,h_out = mse_encoder(X,T,W,V)
    h_out = h_out[:-1]
    #print(h_out)
    h_out = np.sign(h_out)
    print(h_out)
    o_out = np.sign(o_out)
    print(o_out)
    print(error)




def two_layer_train(X, T, W, V, epoch, eta):
    acc_list = []
    iterations = []
    error_list = []
    for i in range(epoch):
        # print(W)
        # print(V)
        """ 
        First we get o_out(final output) and h_out (output from hidden node) with forward pass, 
            then we get the delta_h,delta_o by using back propagation, finally we set deltaW,deltaV and repeat for set epochs
        """

        prev_W, prev_V = W, V
        o_out, h_out = forward_pass(X,W,V)
        delta_h, delta_o = back_pass(o_out,h_out,T,V)
        delta_W, delta_V = get_delta_weights(delta_h, delta_o, X, eta, h_out)
        V, W = update_weights(V,W,delta_W,delta_V)
        #prev_error, new_error = calculate_error(X, prev_W, T), calculate_error(X, W, T)
        #error_list.append(new_error)
        acc_list.append(calc_accuracy(X, W, V, T))
        iterations.append(i)
        #print(calc_accuracy(X,W,V,T))
    return W, V,acc_list, iterations


def plot_all_two_layer(X, W, eta=0.001, iteration=0):
    """
    Func plot_all/6
    @spec plot_all(np.array(), np.array(), boolean, boolean, integer, integer) :: void
        Plots both the perceptron line & both datasets.
        We can find the line due to the following property:
            Wx = 0
            which in our case means: w0 + w1x1 + w2x2 = 0
    """

    plt.scatter(X[0, n:], X[1, n:], color="red")
    plt.scatter(X[0, :n], X[1, :n], color="blue")
    plt.title("Computing two layer perceptron;")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim(top = 1.5, bottom = -1.5)
    x = np.linspace(-3, 3, 200)
    y_list = []
    for w_row in range(len(W)):
        bias = W[w_row][2]
        k = -(bias / W[w_row][1]) / (bias / W[w_row][0])
        m = -bias / W[w_row][1]
        y_list.append(k*x + m)
    # plt3d.plot_surface(xx, yy, z, alpha=0.2)
    for linje in y_list:
        plt.plot(x, linje, color="green")
    plt.show()


def plot_sets(X, eta, plot_validation=False):
    """
    Func plot_all/6
    @spec plot_all(np.array(), np.array(), boolean, boolean, integer, integer) :: void
        Plots both the perceptron line & both datasets.
        We can find the line due to the following property:
            Wx = 0
            which in our case means: w0 + w1x1 + w2x2 = 0
    """
    if plot_validation:
        plt.scatter(X[0, :100], X[1, :100], color="blue")
        plt.scatter(X[0, 100:], X[1, 100:], color="red")
        plt.title(f"Multi-layer perceptron learning {eta}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.ylim(top=0.5, bottom=-0.5)
        plt.show()
    else:
        plt.scatter(X[0, :100], X[1, :100], color="blue")
        plt.scatter(X[0, 100:], X[1, 100:], color="red")
        plt.title(f"Multi-layer perceptron learning {eta}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.ylim(top=0.5, bottom=-0.5)
        plt.show()


def calc_accuracy(X, W, V, T, non_separable = True):
    """
    Func calc_accuracy/3
    @spec calc_accuracy(np.array(), np.array(), np.array()) :: integer
        Counts how many of the input values Xi in X are classified correctly given the weight and target matrices W & T.
    """
    o_out, h_out = forward_pass(X, W, V)
    count = 0
    if non_separable:
        for i in range(len(T[0])):
            if o_out[0][i] > 0 and T[0][i] == 1:
                count += 1
            if o_out[0][i] < 0 and T[0][i] == -1:
                count += 1
    else:
        for i in range(len(T[0])):
            if o_out[0][i] > 0 and T[0][i] == 1:
                count += 1
            if o_out[0][i] < 0 and T[0][i] == -1:
                count += 1
    return count/(len(T))


def plot_accuracy(accuracy_list, iteration_list):
    plt.plot(iteration_list, accuracy_list, color="green")
    plt.xlabel("Number of iterations")
    plt.ylabel("Accuracy rate")
    plt.title(f"2-layer perceptron: {hidden_neurons} hidden neurons")
    plt.show()


def plot_mean_squared_error(error_list, iteration_list):
    plt.plot(iteration_list, error_list, color="green")
    plt.xlabel("Number of iterations")
    plt.ylabel("Mean Squared Error")
    plt.title(f"2-layer perceptron: {hidden_neurons} hidden neurons")
    plt.show()




def split_X(X, ratio):
    """
    Return new_x,  validation_set_x, new_t
    """
    new_X = np.ones([3, 150])
    validation_set_x = np.ones([3, 50])
    target_set_t = np.ones([1, 150])
    if ratio == 1:
        # 25/25
        new_X[0, :75] = X[0, :75]
        new_X[1, :75] = X[1, :75]
        new_X[0, 75:] = X[0, 100:175]
        new_X[1, 75:] = X[1, 100:175]
        validation_set_x[0, :25] = X[0, 75:100]
        validation_set_x[1, :25] = X[1, 75:100]
        validation_set_x[0, 25:] = X[0, 175:]
        validation_set_x[1, 25:] = X[1, 175:]
        target_set_t[0, :75] = -1
        plt.scatter(new_X[0, :75], new_X[1, :75], color="blue")
        plt.scatter(new_X[0, 75:], new_X[1, 75:], color="red")
        plt.show()
        plt.scatter(validation_set_x[0, :25], validation_set_x[1, :25], color="blue")
        plt.scatter(validation_set_x[0, 25:], validation_set_x[1, 25:], color="red")
        plt.show()
        new_target_set = np.ones([1, 50])
        new_target_set[0, :25] = -1
        return new_X, validation_set_x, target_set_t, new_target_set

    elif ratio == 2:
        # 50 A
        new_X[0, :50] = X[0, :50]
        new_X[1, :50] = X[1, :50]
        new_X[0, 50:] = X[0, 100:]
        new_X[1, 50:] = X[1, 100:]
        validation_set_x[0] = X[0, 50:100]
        validation_set_x[1] = X[1, 50:100]
        target_set_t[0, :50] = -1
        plt.scatter(new_X[0, :50], new_X[1, :50], color="blue")
        plt.scatter(new_X[0, 50:], new_X[1, 50:], color="red")
        plt.show()
        plt.scatter(validation_set_x[0, :50], validation_set_x[1, :50], color="blue")
        plt.show()
        new_target_set = np.ones([1, 50])
        new_target_set[0, :50] = -1
        return new_X, validation_set_x, target_set_t, new_target_set

    elif ratio == 3:
        # 50 B
        new_X[0, :100] = X[0, :100]
        new_X[1, :100] = X[1, :100]
        new_X[0, 100:] = X[0, 100:150]
        new_X[1, 100:] = X[1, 100:150]
        validation_set_x[0] = X[0, 150:]
        validation_set_x[1] = X[1, 150:]
        target_set_t[0, :100] = -1
        plt.scatter(new_X[0, :100], new_X[1, :100], color="blue")
        plt.scatter(new_X[0, 100:], new_X[1, 100:], color="red")
        plt.show()
        plt.scatter(validation_set_x[0, :50], validation_set_x[1, :50], color="red")
        plt.show()
        new_target_set = np.ones([1, 50])
        return new_X, validation_set_x, target_set_t, new_target_set

    elif ratio == 4:
        # 20 < 0 80 > 0 A
        new_X[0, :40] = X[0, :40]
        new_X[0, 40:50] = X[0, 50:60]
        new_X[1, :40] = X[1, :40]
        new_X[1, 40:50] = X[1, 50:60]
        new_X[0, 50:] = X[0, 100:]
        new_X[1, 50:] = X[1, 100:]

        validation_set_x[0, :10] = X[0, 40:50]
        validation_set_x[0, 10:50] = X[0, 60:100]
        validation_set_x[1, :10] = X[1, 40:50]
        validation_set_x[1, 10:50] = X[1, 60:100]
        target_set_t[0, :50] = -1
        plt.scatter(new_X[0, :50], new_X[1, :50], color="blue")
        plt.scatter(new_X[0, 50:], new_X[1, 50:], color="red")
        plt.show()
        plt.scatter(validation_set_x[0, :50], validation_set_x[1, :50], color="blue")
        plt.show()
        new_target_set = np.ones([1, 50])
        new_target_set[0, :50] = -1
        return new_X, validation_set_x, target_set_t, new_target_set
    else:
        print("xd")

"""
    Ratio1: 25/25
    Ratio2: 50A
    Ratio3: 50B
    Ratio4: 20 <0 80 > 0 A
"""
#auto_encode(500000,0.001)
guass_func()
#X, W, V, T = generate_matrices()
#plot_all_two_layer(X, W, learning_rate)
#new_X, validation_X, new_T, validation_T = split_X(X, 1)
#W, V, err_list, acc_list, it_list = two_layer_train(X, T, W, V, 1000, learning_rate)
#plot_all_two_layer(X, W, learning_rate)
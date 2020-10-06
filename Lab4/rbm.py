from util import *

class RestrictedBoltzmannMachine():
    '''
    For more details : A Practical Guide to Training Restricted Boltzmann Machines https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf
    '''
    def __init__(self, ndim_visible, ndim_hidden, is_bottom=False, image_size=[28,28], is_top=False, n_labels=10, batch_size=10):

        """
        Args:
          ndim_visible: Number of units in visible layer.
          ndim_hidden: Number of units in hidden layer.
          is_bottom: True only if this rbm is at the bottom of the stack in a deep belief net. Used to interpret visible layer as image data with dimensions "image_size".
          image_size: Image dimension for visible layer.
          is_top: True only if this rbm is at the top of stack in deep beleif net. Used to interpret visible layer as concatenated with "n_label" unit of label data at the end. 
          n_label: Number of label categories.
          batch_size: Size of mini-batch.
        """
       
        self.ndim_visible = ndim_visible

        self.ndim_hidden = ndim_hidden

        self.is_bottom = is_bottom

        if is_bottom : self.image_size = image_size
        
        self.is_top = is_top

        if is_top : self.n_labels = 10

        self.batch_size = batch_size        
                
        self.delta_bias_v = 0

        self.delta_weight_vh = 0

        self.delta_bias_h = 0

        self.bias_v = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible))

        self.weight_vh = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible,self.ndim_hidden))

        self.bias_h = np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_hidden))
        
        self.delta_weight_v_to_h = 0

        self.delta_weight_h_to_v = 0        
        
        self.weight_v_to_h = None
        
        self.weight_h_to_v = None

        self.learning_rate = 0.01
        
        self.momentum = 0.7

        self.print_period = 100
        
        self.rf = { # receptive-fields. Only applicable when visible layer is input data
            "period" : 5000, # iteration period to visualize
            "grid" : [5,5], # size of the grid
            "ids" : np.random.randint(0,self.ndim_hidden,25) # pick some random hidden units
            }
        
        return

        
    def cd1(self,visible_trainset, n_iterations=10000):
        
        """Contrastive Divergence with k=1 full alternating Gibbs sampling

        Args:
          visible_trainset: training data for this rbm, shape is (size of training set, size of visible layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print ("learning CD1")
        
        n_samples = visible_trainset.shape[0]
        swifty_swipe = int(n_samples/self.batch_size)
        recon_loss = []
        for epoch in range(n_iterations):
            print(f"Epoch number {epoch + 1} - lets go!")
            for it in range(swifty_swipe):
                start_indx = it * self.batch_size
                end_indx = (it + 1) * self.batch_size
                minibatch = visible_trainset[start_indx:end_indx, :]
                # [TODO TASK 4.1] run k=1 alternating Gibbs sampling : v_0 -> h_0 ->  v_1 -> h_1.
                # you may need to use the inference functions 'get_h_given_v' and 'get_v_given_h'.
                # note that inference methods returns both probabilities and activations (samples from probablities) and you may have to decide when to use what.
                v_0 = minibatch

                # Positive phase
                _, h_0 = self.get_h_given_v(v_0)

                # Negative phase
                _, v_k = self.get_v_given_h(h_0)
                _, h_k = self.get_h_given_v(v_k)
                # [TODO TASK 4.1] update the parameters using function 'update_params'
                self.update_params(v_0, h_0, v_k, h_k)
            # visualize every epoch when visible layer is input images
            # viz_rf(weights=self.weight_vh[:,self.rf["ids"]].reshape((self.image_size[0],self.image_size[1],-1)), it=epoch, grid=self.rf["grid"])

            # Calculate recon loss for cur epoch
            _, h_0temp = self.get_h_given_v(visible_trainset)
            _, v_ktemp = self.get_v_given_h(h_0temp)
            r_loss = np.sum(((visible_trainset - v_ktemp)**2)/n_samples)
            # print progress
            print("epoch=%7d recon_loss MSE=%4.4f" % (epoch + 1, r_loss))
            recon_loss.append(r_loss)
        return recon_loss
    

    def update_params(self,v_0,h_0,v_k,h_k):

        """Update the weight and bias parameters.

        You could also add weight decay and momentum for weight updates.

        Args:
           v_0: activities or probabilities of visible layer (data to the rbm)
           h_0: activities or probabilities of hidden layer
           v_k: activities or probabilities of visible layer
           h_k: activities or probabilities of hidden layer
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.1] get the gradients from the arguments (replace the 0s below) and update the weight and bias parameters

        # Positive phase - go from v to h
        pos_v = np.sum(v_0, axis=0)
        pos_h = np.sum(h_0, axis=0)

        # NEgative phase - go from h back to v
        neg_v = np.sum(v_k, axis=0)
        neg_h = np.sum(h_k, axis=0)

        # Take difference of pos and neg v-probabilities
        # Utilize regulation technique - 'momentum' to avoid overflow please?
        self.delta_bias_v = self.momentum * self.delta_bias_v + (self.learning_rate/self.batch_size) * (pos_v - neg_v)

        # Diff of log probability between cd0 and cd1 - basically gibbs sampling
        self.delta_weight_vh = self.momentum * self.delta_weight_vh + (self.learning_rate/self.batch_size) * (np.dot(v_0.T, h_0) - np.dot(v_k.T, h_k))

        # Take difference of pos and neg h-probabilities
        self.delta_bias_h = self.momentum * self.delta_bias_h + (self.learning_rate/self.batch_size) * (pos_h - neg_h)
        
        self.bias_v += self.delta_bias_v
        self.weight_vh += self.delta_weight_vh
        self.bias_h += self.delta_bias_h
        
        return

    def get_h_given_v(self, visible_minibatch):
        
        """Compute probabilities p(h|v) and activations h ~ p(h|v) 

        Uses undirected weight "weight_vh" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """
        
        assert self.weight_vh is not None

        n_samples = visible_minibatch.shape[0]

        # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of hidden layer (replace the zeros below) 
        prob = sigmoid(self.bias_h + np.dot(visible_minibatch, self.weight_vh))
        activated = sample_binary(prob)
        return prob, activated


    def get_v_given_h(self,hidden_minibatch):
        
        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses undirected weight "weight_vh" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """
        
        assert self.weight_vh is not None

        n_samples = hidden_minibatch.shape[0]

        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """

            # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass below). \
            # Note that this section can also be postponed until TASK 4.2, since in this task, stand-alone RBMs do not contain labels in visible layer.

            # Get the probabilities - negative phase, vi går från hidden till visible det är därför vi behöver self.bias_v
            support = self.bias_v + np.dot(hidden_minibatch, self.weight_vh.T)

            # Threshold support
            support[support < -75] = -75

            # Retrieve the labels:
            #       first -> all labels from 0 -> len(support[0] - n_labels)
            #       second -> all labels from len(support[0] - n_labels) -> n_labels
            data_support, label_support = support[:, :-self.n_labels], support[:, -self.n_labels:]

            # Calculate the probabilites <v_i h_j> given the hidden_minibatch and support data
            probabilities = sigmoid(data_support)

            # Label probabilities with softmax instead of exponential energy function given from sigmoid. USING BIAS!?
            label_probabilities = softmax(label_support)

            samples = sample_binary(probabilities)
            label_samples = sample_categorical(label_probabilities)

            v_prob = np.concatenate((probabilities, label_probabilities), axis=1)
            v_sample = np.concatenate((samples, label_samples), axis=1)

        else:
                        
            # [TODO TASK 4.1] compute probabilities and activations (samples from probabilities) of visible layer (replace the pass and zeros below)
            v_prob = sigmoid(self.bias_v + np.dot(hidden_minibatch, self.weight_vh.T))
            v_sample = sample_binary(v_prob)
        
        return v_prob, v_sample


    
    """ rbm as a belief layer : the functions below do not have to be changed until running a deep belief net """

    def untwine_weights(self):
        
        self.weight_v_to_h = np.copy( self.weight_vh )
        self.weight_h_to_v = np.copy( np.transpose(self.weight_vh) )
        self.weight_vh = None

    def get_h_given_v_dir(self,visible_minibatch):

        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses directed weight "weight_v_to_h" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """
        
        assert self.weight_v_to_h is not None

        n_samples = visible_minibatch.shape[0]

        # [TODO TASK 4.2] perform same computation as the function 'get_h_given_v' but with directed connections (replace the zeros below) 
        h_prob = sigmoid(self.bias_h + np.dot(visible_minibatch, self.weight_v_to_h))
        h_sample = sample_binary(h_prob)
        return h_prob, h_sample


    def get_v_given_h_dir(self,hidden_minibatch):


        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses directed weight "weight_h_to_v" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """
        
        assert self.weight_h_to_v is not None
        
        n_samples = hidden_minibatch.shape[0]
        
        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """
            
            # [TODO TASK 4.2] Note that even though this function performs same computation as 'get_v_given_h' but with directed connections,
            # this case should never be executed : when the RBM is a part of a DBN and is at the top, it will have not have directed connections.
            # Appropriate code here is to raise an error (replace pass below)
            print("uh oh stinky")
            exit()
            
        else:
            # [TODO TASK 4.2] performs same computaton as the function 'get_v_given_h' but with directed connections (replace the pass and zeros below)
            v_prob = sigmoid(self.bias_v + np.dot(hidden_minibatch, self.weight_h_to_v))
            v_sample = sample_binary(v_prob)
            return v_prob, v_sample


    # DO NOT DO BELOW YET
    def update_generate_params(self,inps,trgs,preds):
        
        """Update generative weight "weight_h_to_v" and bias "bias_v"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.
        
        self.delta_weight_h_to_v += 0
        self.delta_bias_v += 0
        
        self.weight_h_to_v += self.delta_weight_h_to_v
        self.bias_v += self.delta_bias_v 
        
        return
    
    def update_recognize_params(self,inps,trgs,preds):
        
        """Update recognition weight "weight_v_to_h" and bias "bias_h"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.

        self.delta_weight_v_to_h += 0
        self.delta_bias_h += 0

        self.weight_v_to_h += self.delta_weight_v_to_h
        self.bias_h += self.delta_bias_h
        
        return    

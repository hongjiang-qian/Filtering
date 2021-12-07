"""
network.py (mini-batch update learning rate.)
~~~~~~~~~~
Code is based on Michael Nielsen.
"""
#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np
#from numba import jit
from sklearn.metrics import mean_squared_error

class Network(object):

    def __init__(self, sizes, default_initial=True):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        np.random.seed(123)
        if default_initial:
            # He initialization
            # self.biases_initial = [np.random.randn(y, 1)/np.sqrt(1/2) for y in sizes[1:]]
            # self.weights_initial = [np.random.randn(y, x)/np.sqrt(x/2) for x, y in zip(sizes[:-1], sizes[1:])]
            np.random.seed(1122)
            self.biases_initial=[np.random.randn(y, 1) for y in sizes[1:]]
            self.weights_initial=[np.random.randn(y, x)/np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]
        else:
            np.random.seed(112233)
            self.biases_initial = [np.random.randn(y, 1) for y in sizes[1:]]
            self.weights_initial = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.eta=0

    def params_initializer(self):
        self.biases = self.biases_initial
        self.weights = self.weights_initial
        
    def feedforward(self, a, weights=None, biases=None):
        """Return the output of the network if ``a`` is input."""
        if weights is None:
            weights=self.weights
        if biases is None:
            biases=self.biases
        #!!! Note the output layer has no activation for regression.
        for b, w in zip(biases[:-1], weights[:-1]):
            a = relu(np.dot(w, a)+b)
        # last layer for regression.
        a=np.dot(weights[-1], a)+biases[-1]
        return a
    
    def SGD_Constlr(self, training_data, epochs, mini_batch_size, learning_rate, seed=True, evaluation_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""

        """Mini batch stochastic gradient descent with constant learning rate,
        which can be viewed as default optimizer for SGD."""
        print("-----------------------------")
        print("Constant learning rate begins.")
        self.params_initializer()
        self.eta=learning_rate

        print("Initial eta is: {0}".format(learning_rate)) 
        if evaluation_data: n_evaluation = len(evaluation_data)
        n = len(training_data)
        training_cost=[]; evaluation_cost=[]
        training_relerr=[]; evaluation_relerr=[]
        lr=[];lr.append(learning_rate)

        if seed: random.seed(1)
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches=[training_data[k:k+mini_batch_size] for k in range(0,n, mini_batch_size)]
            for mini_batch in mini_batches:
                nabla_b, nabla_w=self.mini_batch_gradients(mini_batch)
                self.update_weights(nabla_b, nabla_w)
            lr.append(self.eta)
            print("The eta of Epoch {0} is {1}".format(epoch, self.eta))
            tc_mseErr, tc_relErr=self.total_cost(training_data)
            ec_mseErr, ec_relErr=self.total_cost(evaluation_data)
            training_cost.append(tc_mseErr); training_relerr.append(tc_relErr)
            evaluation_cost.append(ec_mseErr); evaluation_relerr.append(ec_mseErr)

            if evaluation_data:
                print("Epoch {0}: {1}".format(epoch, evaluation_cost[-1]))
            else:
                print ("Epoch {0} complete".format(epoch))

        return training_cost, training_relerr, evaluation_cost, evaluation_relerr, self.weights, self.biases, lr

    def SGD_EpochAdalr(self, training_data, epochs, mini_batch_size, learning_rate, epi, alpha,\
                       modified=False, monitor_cost=True, seed=True, evaluation_data=None):
        """Mini_batch Stochastic gradient descent with adaptive lr based on Epoch. That is to say 
        the learning rate will change every epoch. Modified is true means we compute learning rate gradient
        by one side pertubation rather than two-sided."""
        print("-----------------------------")
        print("Epoch Adaptive lr begins.")
        self.params_initializer()
        self.eta=learning_rate
        print("Initial eta is: {0}".format(learning_rate))

        if evaluation_data: n_evaluation = len(evaluation_data)
        n = len(training_data)
        training_cost=[]; evaluation_cost=[]
        training_relerr=[]; evaluation_relerr=[]
        lr=[];lr.append(learning_rate)
        if seed: random.seed(3)
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                nabla_b, nabla_w=self.mini_batch_gradients(mini_batch)
                self.update_weights(nabla_b, nabla_w)
            # Now we update learning rate
            # First to compute gradients on whole training data, then perturb it to get nabla_eta.
            nabla_b_total, nabla_w_total=self.mini_batch_gradients(training_data)
            if modified:
                nabla_eta=self.learning_rate_gradient(training_data, nabla_b_total, nabla_w_total, epi, alpha, modified=True)
            else:
                nabla_eta=self.learning_rate_gradient(training_data, nabla_b_total, nabla_w_total, epi, alpha)
            self.eta=self.eta-epi*nabla_eta; lr.append(self.eta)
            if monitor_cost:
                tc_mseErr, tc_relErr=self.total_cost(training_data)
                ec_mseErr, ec_relErr=self.total_cost(evaluation_data)
                training_cost.append(tc_mseErr); training_relerr.append(tc_relErr)
                evaluation_cost.append(ec_mseErr); evaluation_relerr.append(ec_mseErr)
            print("The eta of Epoch {0} is {1}".format(epoch, self.eta))
            if (monitor_cost and evaluation_data):
                print("Epoch {0}: {1}".format(epoch, evaluation_cost[-1]))
            else:
                print ("Epoch {0} complete".format(epoch))
                
        return training_cost, training_relerr, evaluation_cost, evaluation_relerr, self.weights, self.biases, lr

    def mini_batch_gradients(self, mini_batch):
        """Return the gradient of the mini_batch."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w= self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # note to divide the length of mini_batch.    
        nabla_b=[nb/len(mini_batch) for nb in nabla_b]
        nabla_w=[nw/len(mini_batch) for nw in nabla_w]
        return (nabla_b, nabla_w)
    
    def learning_rate_gradient(self, mini_batch, nabla_b, nabla_w, epi, alpha, modified=False):
        """Return the estimation of the gradient of another cost function
        with respect to learning rate. 'epi' is the updating stepsize of learning rate
        , 'alpha' is the perturbation length of the current learning rate.
        'modified' concerns whether to perturb on both sides. It is false means that we
        perturb the learning rate to the right and to the left."""

        # perturb weights to right and left.
        weights_right=[w-(self.eta+alpha)*nw for w, nw in zip(self.weights, nabla_w)]
        biases_right=[b-(self.eta+alpha)*nb for b, nb in zip(self.biases, nabla_b)]
        loss_right=0
        if modified:
            weights_cur=[w-self.eta*nw for w, nw in zip(self.weights,nabla_w)]
            biases_cur=[b-self.eta*nb for b, nb in zip(self.biases, nabla_b)]
            loss_cur=0
        else:
            weights_left=[w-(self.eta-alpha)*nw for w, nw in zip(self.weights,nabla_w)]
            biases_left=[b-(self.eta-alpha)*nb for b, nb in zip(self.biases, nabla_b)]
            loss_left=0

        for x, y in mini_batch:
            loss_right+=self.QuadraticCost(self.feedforward(x, weights_right, biases_right), y)
            if modified:
                loss_cur+=self.QuadraticCost(self.feedforward(x, weights_cur, biases_cur), y)
            else:
                loss_left+=self.QuadraticCost(self.feedforward(x, weights_left, biases_left), y)
        loss_right=loss_right/len(mini_batch)
        if modified:
            loss_cur=loss_cur/len(mini_batch)
            nabla_eta=(loss_right-loss_cur)/alpha
        else:
            loss_left=loss_left/len(mini_batch)
            nabla_eta=(loss_right-loss_left)/(2*alpha)
        #print("Loss perturbed is: {0}".format(loss_right-loss_left))
        #print("nabla_eta is:{0}".format(nabla_eta))
        return nabla_eta

    def update_weights(self, nabla_b, nabla_w, eta=None):
        if eta is None:
            eta=self.eta
        self.weights=[w-eta*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases=[b-eta*nb for b, nb in zip(self.biases, nabla_b)]        
    
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # forward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = relu(z)
            activations.append(activation)
        #===== Regression =======
        # deal with last layer (no activation)
        z=np.dot(self.weights[-1],activation)+self.biases[-1]
        zs.append(z)
        activation=z # no activation.
        activations.append(activation)

        # backward pass
        #====== Michael Nielsen version =========
        #delta = self.cost_derivative(activations[-1], y) * \
        #    sigmoid_prime(zs[-1])
        #nabla_b[-1] = delta
        #nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        #====== Regresion Version ==========
        # the last year has no activation, it could be viewed as activation
        # with x, i.e. sigmoid_prime=1 or [1,1,1...,1].
        delta = self.cost_derivative(activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative index in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = relu_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    #===== Michael Nielsen version: quadratic loss ======
    #def cost(self,output_activations,y):
    #    return(np.square(np.linalg.norm(output_activations-y))/2)

    #def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives partial C_x /
    #    partial a for the output activations."""
    #    return (output_activations-y)

    #=====  MSE version =====
    def cost(self, output_activations, y):
        return mean_squared_error(output_activations, y)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives partial C_x/ partial a
        for the output_activeations."""
        return (2/len(y))*(output_activations-y)

    def total_cost(self, data, weights=None, biases=None, convert=False):
        """Evaluation data needs to be converted."""
        mse_err=0.0; rel_err=0.0
        if weights is None:
            weights=self.weights
        if biases is None:
            biases=self.biases
        for x, y in data:
            a=self.feedforward(x, weights,biases)
            if convert: y=vectorize(y)
            mse_err+=self.cost(a,y)
            rel_err+=self.relative_err(a,y)
        mse_err=mse_err/len(data)
        rel_err=rel_err/len(data)
        return (mse_err, rel_err)

    def QuadraticCost(self, output_activations, y):
        return 0.5*np.linalg.norm(output_activations-y)
        
    def predict(self,data, weights=None, biases=None,label=False):
        if weights is None:
            weights=self.weights
        if biases is None:
            biases=self.biases
        if label:
            pred=[self.feedforward(x, weights, biases) for x, y in data]
        pred=[self.feedforward(x, weights, biases) for x in data]
        return pred

    def relative_err(self, y_true, y_pred):
        """Compute Relative Error"""
        abs_sum=np.sum(np.abs(y_true-y_pred))
        return abs_sum/(np.sum(np.abs(y_true))+np.sum(np.abs(y_pred)))

#### Miscellaneous functions

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def vectorize(y):
    """Return a 10 dimensional unit vector with 1.0 at j-th position."""
    e = np.zeros((10, 1))
    e[y] = 1.0
    return e
    
def relu(z):
    z[z<=0]=0
    return z

def relu_prime(z):
    z[z>0]=1
    z[z<0]=0
    return z
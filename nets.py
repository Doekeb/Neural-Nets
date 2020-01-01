import sys
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import itertools as iter
rd = np.random
import plotting
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.widgets import Slider
from copy import deepcopy
from matplotlib import animation


activators = {'sigmoid': '1/(1 + e^(-z))',
              'arctan': 'arctan(z)/pi + 1/2',
              'exponential': '1/2 e^z (if z <= 0), 1 - 1/2 e^(-z) (if z >= 0)',
              'tanh': '1/2 (tanh(z) + 1)',
              'relu': '0 (if z <= 0), z (if z >= 0)',
              'lrelu': '0.01 z (if z <= 0), z (if z >= 0)',
              'linear': 'z'}

def sigmoid(z):
    """
    Sigmoidal activator function (between 0 and 1)
    """
    return 1/(1+np.exp(-z))

def d_sigmoid(z):
    """
    Derivative of the sigmoidal activator
    """
    return sigmoid(z)*(1-sigmoid(z))

def arctan(z):
    """
    (Shifted and scaled) arc-tangent activator function (between 0 and 1)
    """
    return np.arctan(z)/np.pi + np.float64(1)/2

def d_arctan(z):
    """
    Derivative of the arc-tangent activator
    """
    return 1/(np.pi*(1+z**2))

def exponential(z):
    """
    Piecewise exponential activator function (between 0 and 1)
    """
    if z <= 0:
        return 1/2*np.exp(z)
    if z >= 0:
        return 1-1/2*np.exp(-z)

def d_exponential(z):
    """
    Derivative of the exponential activator
    """
    if z <= 0:
        return 1/2*np.exp(z)
    if z >= 0:
        return 1/2*np.exp(-z)

def tanh(z):
    """
    (Shifted) hyperbolic tangent activator function
    """
    return 1/2*(np.tanh(z)+1)

def d_tanh(z):
    """
    Derivative of the hyperbolic tangent activator
    """
    return 1/2*(1 - np.tanh(z)**2)

def relu(z):
    """
    Rectified linear unit activator function
    """
    if z >= 0:
        return z
    if z <= 0:
        return np.float64(0)

def d_relu(z):
    """
    Derivative of the rectified linear unit activator
    """
    if z >= 0:
        return np.float64(1)
    if z <= 0:
        return np.float64(0)

def lrelu(z):
    """
    Leaky rectified linear unit activator function
    """
    if z >= 0:
        return z
    if z <= 0:
        return 0.01*z

def d_lrelu(z):
    """
    Derivative of the leaky rectified linear unit activator
    """
    if z >= 0:
        return np.float64(1)
    if z <= 0:
        return 0.01

def linear(z):
    """
    Linear activator function
    """
    return z

def d_linear(z):
    """
    Derivative of the linear activator
    """
    return np.float64(1)

class Neuron():
    """
    This is a class for a single neuron to be used in a neural network.

    Attributes:
        weights (array of floats): The weights for this neuron's inputs
        bias (float): This neuron's bias
        activator_name (string): The name of this neuron's activator function
        activator (function): This neuron's activator function
        d_activator (function): The derivative of this neuron's activator
    """
    def __init__(self, weights, bias, activator='sigmoid'):
        """
        Create a new Neuron instance.

        Parameters:
            weights (iterable of floats): The weights for this neuron's inputs
            bias (float): This neuron's bias
            activator (string): The name of this neuron's activator function (one of
                'sigmoid' (default), 'arctan', 'exponential', 'tanh', 'relu', 'lrelu',
                'linear')
        """
        self.weights = np.array(weights, np.float64)
        self.bias = np.float64(bias)
        self.activator_name = activator

        # Get the activator function (and derivative) by name
        self.activator = getattr(sys.modules[__name__], activator)
        self.d_activator = getattr(sys.modules[__name__], 'd_' + activator)

    def __repr__(self):
        """
        Return a string representation of this neuron.
        """
        return "A neuron with weights %s, bias %s, and %s activator" \
               % (self.weights, self.bias, self.activator_name)

    def fire(self, inputs, weights=None, bias=None):
        """
        Fire this neuron with a given set of inputs.

        Parameters:
            inputs (iterable of floats): The inputs for which to fire this neuron
            weights (iterable of floats): The weights for the inputs (default: None, in
                which case this neuron's weights are used)
            bias (float): The bias for this neuron (default: None, in which case this
                neuron's bias is used)

        Returns:
            float: The result of firing this neuron
        """
        if weights is None:
            weights = self.weights
        if bias is None:
            bias = self.bias
        self.inputs = np.array(inputs, np.float64)
        self.z = np.dot(inputs, weights) + bias
        self.output = self.activator(self.z)
        return self.output

    def back_prop(self, inputs, rate=0.1):
        """
        Adjust this neuron's weights and bias according to an input

        Parameters:
            inputs (array of floats): The inputs according to which the weights and bias are
                adjusted
            rate (float): The learning rate for this backwards propogation

        Returns:
            float: A value which, in a network, will be passed to neurons in the previous
                layer
        """
        # sigmoid has a nice derivative
        if self.activator_name == 'sigmoid':
            factor = sum(inputs) * self.output * (1-self.output)
        else:
            factor = sum(inputs) * self.d_activator(self.z)
        old_weights = np.array(self.weights) # Make a copy

        # update the weights with respect to learning rate
        self.weights -= rate * factor * self.inputs

        # update the bias with respect to learning rate
        self.bias -= rate * factor

        return factor * old_weights

def sum_of_squares(results, targets):
    """
    Sum of squares loss function

    Parameters:
        results (array of floats): The observed values
        targets (array of floats): The desired values

    Returns:
        float: Measure of error between results and targets
    """
    return sum((results-targets)**2)

def d_sum_of_squares(results, targets):
    """
    Multi-derivative of the sum of squares loss function

    Parameters:
        results (array of floats): The observed values
        targets (array of floats): The desired values

    Returns:
        array of floats: Each entry is the derivative of the loss function with
            respect to the corresponding coordinate (result) variable, evaluated at
            the value of that result variable
    """
    return 2*(results-targets)

def mse(results, targets):
    """
    Mean squared error loss function

    Parameters:
        results (array of floats): The observed values
        targets (array of floats): The desired values

    Returns:
        float: Measure of error between results and targets
    """
    return sum((results-targets)**2) / len(results)

def d_mse(results, targets):
    """
    Multi-derivative of the mean squared error loss function

    Parameters:
        results (array of floats): The observed values
        targets (array of floats): The desired values

    Returns:
        array of floats: Each entry is the derivative of the loss function with
            respect to the corresponding coordinate (result) variable, evaluated at
            the value of that result variable
    """
    return 2*(results-targets) / len(results)

def L2(results, targets):
    """
    L^2 metric loss function

    Parameters:
        results (array of floats): The observed values
        targets (array of floats): The desired values

    Returns:
        float: Measure of error between results and targets
    """
    return np.sqrt(sum((results-targets)**2))

def d_L2(results, targets):
    """
    Multi-derivative of the L^2 metric loss function

    Parameters:
        results (array of floats): The observed values
        targets (array of floats): The desired values

    Returns:
        array of floats: Each entry is the derivative of the loss function with
            respect to the corresponding coordinate (result) variable, evaluated at
            the value of that result variable
    """
    return results / np.sqrt(sum((results-targets)**2))





# activator can be a single string, e.g. 'sigmoid' or a dictionary with keywords
# being neuron types and values the activator string names, or a list of strings
# indicating the activator types per-layer, or a list of lists of strings
# indicating the activator types per-neuron

class NeuralNetwork():
    """
    This is a class for a network of neurons.

    Attributes:
        n_layers (integer): The number of (non-input) layers in this network
        layer_sizes (array of integers): The number of neurons in each layer
        connections (array of matrices): Adjecency matrices between layers
        loss_name (string): The name of the loss function
        loss (function): The loss function
        d_loss (function): The derivative of the loss function
        neurons (array of arrays of Neurons): The list of Neurons at each layer,
            initiallized randomly
        log (False or dict): If False, training data is not logged. Otherwise, it is
            a dictionary consisting of training data with keywords 'loss',
            'weights', and 'biases', and values the corresponding data after each
            training step.
    """
    def __init__(self, layer_sizes, connections=None, activators='sigmoid',
                 log=True, loss='sum_of_squares'):
        """
        Create a new Neural Network instance.

        Parameters:
            layer_sizes (array of integers): The number of neurons in each layer
            connections (array of matrices): Adjecency matrices between layers (if None,
                all neurons in two adjacent layers are connected)
            activators (string | array of strings | dict | array of arrays of strings):
                If a string, the name of the activator function used for all neurons in
                this network.
                If an array of strings, the length of the array should be n_layers, and
                should consist of the names of the activator functions used for neurons
                in the corresponding layers.
                If a dict, the keys should be 'hidden' and 'output' and the values
                should be the (string) names of the activator functions used for the
                corresponding layer types.
                If an array of arrays of strings, the names of the activator functions
                used for each neuron in this network.
                Allowed strings: 'sigmoid' (default), 'arctan', 'exponential', 'tanh',
                'relu', 'lrelu', 'linear'
            log (boolean): Whether or not to log training data
            loss (string): Name of loss function to use (one of 'sum_of_squares', 'mse',
                or 'L2')
        """
        self.n_layers = len(layer_sizes) - 1
        self.layer_sizes = layer_sizes
        if connections == None:
            self.connections = [np.ones((m,n))
                                for (n,m) in zip(layer_sizes, layer_sizes[1:])]
        else:
            self.connections = [np.array(connection)
                                for connection in connections]

        if type(activators) is str:
            activators = [[activators]*n for n in layer_sizes[1:]]
        elif type(activators) is dict:
            activators = [[activators['hidden']]*n for n in layer_sizes[1:-1]] \
                       + [[activators['output']]*layer_sizes[-1]]
        elif type(activators[0]) is str:
            activators = [[a]*n for a, n in zip(activators, layer_sizes)]

        self.loss_name = loss
        self.loss = getattr(sys.modules[__name__], loss)
        self.d_loss = getattr(sys.modules[__name__], 'd_'+loss)
        self.neurons = [[Neuron(k[i] * rd.normal(size=n),
                                rd.normal(),
                                activator=a[i])
                         for i in range(m)]
                         for (n,m,k,a) in zip(layer_sizes,
                                              layer_sizes[1:],
                                              self.connections,
                                              activators)]
        self.log = log and {'loss':[],
                            'weights':[self.get_weights()],
                            'biases':[self.get_biases()]}

    def get_weights(self):
        """
        Return the weights of this network as a list of matrices. Each matrix is the
        weight matrix between two layers.
        """
        return [self.connections[i] * np.array([N.weights
                                                for N in self.neurons[i]])
                for i in range(self.n_layers)]

    def get_biases(self):
        """
        Return the weights of this network as a list of arrays. Each array is the bias
        array of a layer.
        """
        return [np.array([N.bias for N in self.neurons[i]])
                for i in range(self.n_layers)]

    def reset(self):
        """
        Reset the data log (if it exists), and randomize the weights and biases of the
        neurons.
        """
        if self.log:
            self.log = True
        self.__init__(self.layer_sizes, connections=self.connections,
                      activator=self.activator_name, log=self.log,
                      loss=self.loss_name)

    def fire(self, inputs, weights=None, biases=None):
        """
        Fire this network with a given set of inputs.

        Parameters:                                                                     ''
            inputs (iterable of floats): The inputs for which to fire this network
            weights (list of matrices of floats): The weight matrices to use for the
                inputs (default: None, in which case the neuron weights are used)
            biases (float): The bias for this neuron (default: None, in which case the
                neuron biases are used)

        Returns:
            array of floats: The result of firing this network
        """
        results = np.array(inputs)
        if weights is None and biases is None:
            for c_matrix, n_list in zip(self.connections, self.neurons):
                results = np.array([N.fire(row*results)
                                    for (row, N) in zip(c_matrix, n_list)])
            return np.array(results)

        elif weights is None and biases is not None:
            for c_matrix, n_list, b_array in zip(self.connections,
                                                 self.neurons,
                                                 biases):
                results = np.array([N.fire(row*results, bias=b)
                                    for (row, N, b) in zip(c_matrix, n_list,
                                                           b_array)])
            return np.array(results)

        elif weights is not None and biases is None:
            for c_matrix, n_list, w_matrix in zip(self.connections,
                                                  self.neurons,
                                                  weights):
                results = np.array([N.fire(row*results, weights=w)
                                    for (row, N, w) in zip(c_matrix, n_list,
                                                           w_matrix)])
            return np.array(results)

        else:
            for c_matrix, n_list, w_matrix, b_array in zip(self.connections,
                                                           self.neurons,
                                                           weights, biases):
                results = np.array([N.fire(row*results, weights=w, bias=b)
                                    for (row, N, w, b) in zip(c_matrix, n_list,
                                                              w_matrix,
                                                              b_array)])
            return np.array(results)

    def back_prop(self, inputs, rate):
        """
        Adjust the weights and biases of this network's neurons according to an input

        Parameters:
            inputs (array of floats): The inputs according to which the weights and bias are
                adjusted
            rate (float): The learning rate for this backwards propogation
        """
        results = np.diag(inputs)
        for n_list in reversed(self.neurons):
            results = np.array([N.back_prop(result)
                                for (N, result) in zip(n_list, results.T)])

    def test(self, data):
        """
        Test this network against a data set, print the results.

        Parameters:
            data (dictionary): Keys are inputs for the network, values are expected
                results
        """
        for (key, value) in data.items():
            result = self.fire(np.array(key, np.float64))
            print("Input: %s, Output: %s, Expected: %s" % (key, result, value))

    def train(self, data, rate=0.1, batch_size=1, epochs=1):
        """
        Train this network using a data set.

        Parameters:
            data (dictionary): Keys are inputs for the network, values are expected
                results
            rate (float): The training rate
            batch_size (integer): The number of input samples to fire before updating
                the weights and biases
            epochs (integer): The number of times to iterate through the training set
        """
        for _ in range(epochs): # Repeat for how many epochs there are

            # Get the data and randomize it
            items = list(data.items())
            rd.shuffle(items)

            # Form the batches
            n_leftovers = len(data) % batch_size
            if n_leftovers == 0:
                batches = [items[batch_size*i:batch_size*(i+1)]
                           for i in range(len(items) // batch_size)]
            else:
                batches = [items[batch_size*i:batch_size*(i+1)]
                           for i in range(len(items) // batch_size)] \
                        + [items[-n_leftovers:]]

            # Adjust weights once for each batch
            for batch in batches:
                loss_sum = 0
                d_loss_sums = np.zeros(self.layer_sizes[-1])

                # Fire the network once for each sample
                for (key, targets) in batch:
                    results = self.fire(np.array(key))
                    loss_sum += self.loss(results, targets)
                    d_loss_sums += self.d_loss(results, targets)

                # Average the loss and its derivative
                loss_average = loss_sum / len(batch)
                d_loss_averages = d_loss_sums / len(batch)

                # Back-propogate with loss average
                self.back_prop(d_loss_averages, rate)
                # self.neuron.back_prop(np.array([d_loss_average]), rate=rate)

                if self.log: # Log result
                    self.log['loss'] += [loss_average]
                    self.log['weights'] += [self.get_weights()]
                    self.log['biases'] += [self.get_biases()]

    def plot_loss(self):
        """
        Plot the (moving average) loss function over this network's training life
        """
        log = np.array(self.log['loss'])
        L = len(log)
        plotting.plot_moving_average(log, window=int(np.ceil(L/100)))





    # def animate_plot(self, max_frames=):
    #     """
    #     FILL ME IN
    #     """
    #     L, step_size, weights, biases = \
    #                             self._get_filtered_data(max_frames=max_frames)
    #
    #     # X = np.arange(0, 1+coarseness, coarseness)
    #     # data = np.array([[self.fire([x], weights=ws, biases=bs)[0] for x in X] for ws, bs in zip(weights,biases)])
    #     #
    #     # return (L, step_size, data)

    def _plot(self, weights=None, biases=None, ax=None):
        """
        Set up a graph representing this network.

        Parameters:
            weights (list of matrices): The weight matrices to use for plotting
                (default: None, in which case the neuron weights are used)
            biases (list of arrays): The bias arrays to use for plotting (default: None,
                in which case the neuron biases are used)
        """
        if weights is None:
            weights = self.get_weights()
        if biases is None:
            biases = self.get_biases()

        neurons = self.neurons
        v_scale = 1
        height = v_scale * max(self.layer_sizes)
        h_scale = 1
        width = h_scale * (self.n_layers + 1)

        pos = {}
        labels = {}

        G = nx.DiGraph()
        n_layers = self.n_layers + 1
        shapes = ['<', 'v', '^', 'd', '8', 'h', 'p', 's', 'o']
        activator_shapes = {'input': '>'}
        for i in range(n_layers):
            n_nodes = self.layer_sizes[i]
            for n in range(n_nodes):
                node = (i,n)
                G.add_node(node)
                position = np.array([width / n_layers * i,
                                     height - height / (n_nodes+1) * (n+1)])
                pos[node] = position
                if i == 0:
                    labels[node] = ""
                    color = (0.0, 0.0, 0.0, 1.0)
                    shape = '>'
                else:
                    bias = biases[i-1][n]
                    labels[node] = str(bias)
                    color = (1-sigmoid(bias), 0.0, sigmoid(bias),
                             abs(2*(sigmoid(bias)-1/2)))
                    activator_name = neurons[i-1][n].activator_name
                    try:
                        shape = activator_shapes[activator_name]
                    except KeyError:
                        try:
                            shape = shapes.pop()
                            activator_shapes = {activator_name: shape}
                        except IndexError:
                            shape = '<'
                            activator_shapes = {activator_name: shape}

                nx.draw_networkx_nodes(G,
                                       pos={node:position},
                                       nodelist=[node],
                                       node_color=[color],
                                       edgecolors=[(0.0, 0.0, 0.0, 1.0)],
                                       node_shape=shape,
                                       ax=ax)


        max_thickness = 20
        for i in range(self.n_layers):
            for n in range(self.layer_sizes[i]):
                for m in range(self.layer_sizes[i+1]):
                    # print([((i,n), (i+1,m))])
                    # print(weights[i])
                    weight = weights[i][m][n]
                    edge = ((i,n), (i+1,m), {'weight':weight})
                    G.add_edges_from([edge])
                    alpha = abs(2*(sigmoid(weight)-1/2))
                    color = (1.0, 0.0, 0.0, alpha) if np.sign(weight) == 1 \
                            else (0.0, 0.0, 1.0, alpha)
                    width = max_thickness * alpha
                    nx.draw_networkx_edges(G,
                                           pos=pos,
                                           edgelist=[edge],
                                           width=width,
                                           edge_color=color,
                                           arrows=False,
                                           ax=ax)

    def plot(self, weights=None, biases=None):
        """
        Show a graph representing this network.

        Parameters:
            weights (list of matrices): The weight matrices to use for plotting
                (default: None, in which case the neuron weights are used)
            biases (list of arrays): The bias arrays to use for plotting (default: None,
                in which case the neuron biases are used)
        """
        self._plot(weights=weights, biases=biases)
        plt.show()




    def plot_animation(self, max_frames=200):
        """
        Animate the graph representing this network over its training life.

        Parameters:
            max_frames (integer): The maximum number of frames to use for the animation
        """
        def update(num):
            ax.cla()
            self._plot(weights=weights[num], biases=biases[num])

        # def set_axes():
        #     ax.set_xlim(0.0, 1.0)
        #     ax.set_ylim(0.0, 1.0)

        L, step_size, weights, biases = \
                                self._get_filtered_data(max_frames=max_frames)

        fig = plt.figure()
        ax = fig.gca()
        self._plot(weights=weights[0], biases=biases[0])

        ani = animation.FuncAnimation(fig, update, len(weights), interval=50)

        # return ani

        plt.show()

    def plot_slider(self, max_frames=200):
        """
        Show a graph representing this network with a slider ranging over its training
        life.

        Parameters:
            max_frames (integer): The maximum number of frames to use for the animation
        """
        def update(num):
            ax.cla()
            self._plot(weights=weights[int(num//step_size)], biases=biases[int(num//step_size)], ax=ax)

        # def set_axes():
        #     ax.set_xlim(0.0, 1.0)
        #     ax.set_ylim(0.0, 1.0)

        L, step_size, weights, biases = \
                                self._get_filtered_data(max_frames=max_frames)

        fig = plt.figure()
        ax = fig.gca()
        self._plot(weights=weights[0], biases=biases[0])

        axslider = plt.axes([0.03, 0.1, 0.03, 0.65])
        slider = Slider(axslider, 'Time', 0, L-1, valinit=0, valstep=step_size,
                        orientation='vertical')
        slider.on_changed(update)

        plt.show()


    def _get_dimensions(self):
        """
        Return the dimensions of the input and output vectors for this network.
        """
        return (self.layer_sizes[0], self.layer_sizes[-1])

    def _check_dimensions(self):
        """
        Check that the input and ouput dimensions are appropriate sizes to plot. Raise
        an error if not.
        """
        if self._get_dimensions() not in ((1,1), (1,2), (2,1)):
            msg = '(Input, Output) dimensions must be one of %s, %s, or %s.' \
                  %((1,1), (2,1), (1,2))
            raise(ValueError(msg))

    def _plot_function_11(self, coarseness):
        """
        Plot the function represented by this network, assuming inputs and outputs are
        both one-dimensional.

        Parameters:
            coarseness (float): The distance between sample points
        """
        X = np.arange(0, 1+coarseness, coarseness)
        Y = [self.fire(x)[0] for x in X]
        ax = plt.gca()
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.plot(X,Y)
        plt.show()

    def _plot_function_21(self, coarseness):
        """
        Plot the function represented by this network, assuming inputs are 2-dimensional
        and outputs are 1-dimensional.

        Parameters:
            coarseness (float): The distance between sample points
        """
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        ax.set_zlim(0.0, 1.0)

        X = np.arange(0, 1+coarseness, coarseness)
        Y = np.arange(0, 1+coarseness, coarseness)
        X, Y = np.meshgrid(X, Y)

        Z = []
        for xrow, yrow in zip(X, Y):
            row = []
            for x, y in zip(xrow, yrow):
                row += [self.fire(np.array([x,y]))[0]]
            Z += [row]
        Z = np.array(Z)

        surf = ax.plot_surface(X,Y,Z)
        plt.show()

    def _plot_function_12(self, coarseness):
        """
        Plot the function represented by this network, assuming inputs are 1-dimensional
        and outputs are 2-dimensional.

        Parameters:
            coarseness (float): The distance between sample points
        """
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_zlim(0.0, 1.0)

        X = np.arange(0, 1+coarseness, coarseness)
        results = [self.fire([x]) for x in X]
        Y = np.array([result[0] for result in results])
        Z = np.array([result[1] for result in results])

        ax.plot(X,Y,Z)

        plt.show()

    def plot_function(self, coarseness=None):
        """
        Plot (if possible) the function represented by this network.

        Parameters:
            coarseness (float): The distance between sample points (default is 0.001
                for 1-d to 1-d functions, and 0.01 for all other functions)
        """
        self._check_dimensions()
        a,b = self._get_dimensions()
        if coarseness == None:
            if (a,b) in ((1,1), (1,2)):
                coarseness = 0.001
            else:
                coarseness = 0.01
        f = getattr(self, '_plot_function_%s%s' % (a,b))
        f(coarseness)


    def _get_filtered_data(self, max_frames):
        """
        Return filter data from the training log.

        Parameters:
            max_frames (integer): The maximum number of frames to use for animations

        Returns:
            4-tuple: (length of log,
                      step size required to achieve desired number of frames,
                      weight data filtered to max_frames,
                      bias data filtered to max_frames)
        """
        L = len(self.log['weights'])
        step_size = L//max_frames
        if L > max_frames:
            weights = self.log['weights'][::step_size]
            biases = self.log['biases'][::step_size]
        else:
            weights = self.log['weights']
            biases = self.log['biases']

        return (L, step_size, weights, biases)

    def _setup_11(self, max_frames, coarseness):
        """
        Prepare animation and slider data for 1-d to 1-d functions.

        Parameters:
            max_frames (integer): The maximum number of frames to use for animations
            coarseness (float): The distance between sample points

        Returns:
            4-tuple: (length of log,
                      step size required to achieve desired number of frames,
                      output data frames,
                      input data (same for each frame))
        """
        L, step_size, weights, biases = \
                                self._get_filtered_data(max_frames=max_frames)

        X = np.arange(0, 1+coarseness, coarseness)
        data = np.array([[self.fire([x], weights=ws, biases=bs)[0] for x in X]
                         for ws, bs in zip(weights,biases)])

        return (L, step_size, data, X)

    def _setup_21(self, max_frames, coarseness):
        """
        Prepare animation and slider data for 2-d to 1-d functions.

        Parameters:
            max_frames (integer): The maximum number of frames to use for animations
            coarseness (float): The distance between sample points

        Returns:
            5-tuple: (length of log,
                      step size required to achieve desired number of frames,
                      output data frames,
                      input data for first variable,
                      input data for second variable)
        """
        L, step_size, weights, biases = \
                                self._get_filtered_data(max_frames=max_frames)

        X = np.arange(0, 1+coarseness, coarseness)
        Y = np.arange(0, 1+coarseness, coarseness)
        data = np.array([[[self.fire([x,y], weights=ws, biases=bs)[0]
                           for x in X]
                          for y in Y]
                         for ws, bs in zip(weights, biases)])
        X, Y = np.meshgrid(X,Y)

        return (L, step_size, data, X, Y)

    def _setup_12(self, max_frames, coarseness):
        """
        Prepare animation and slider data for 1-d to 2-d functions.

        Parameters:
            max_frames (integer): The maximum number of frames to use for animations
            coarseness (float): The distance between sample points

        Returns:
        5-tuple: (length of log,
                  step size required to achieve desired number of frames,
                  input data (same for each frame),
                  output data frames for first output variable,
                  output data frames for second output variable)
        """
        L, step_size, weights, biases = \
                                self._get_filtered_data(max_frames=max_frames)

        X = np.arange(0, 1+coarseness, coarseness)
        data = np.array([[self.fire([x], weights=ws, biases=bs) for x in X]
                         for ws, bs in zip(weights, biases)])
        Ys = np.array([[result[0] for result in datum] for datum in data])
        Zs = np.array([[result[1] for result in datum] for datum in data])

        return (L, step_size, X, Ys, Zs)






    def _animate_11(self, max_frames, coarseness):
        """
        Animate the function represented by this network over its training life,
        assuming inputs and outputs are both 1-dimensional.

        Parameters:
            max_frames (integer): The maximum number of frames to use for animations
            coarseness (float): The distance between sample points
        """
        def update(num):
            ax.cla()
            set_axes()
            line = ax.plot(X, data[num])

        def set_axes():
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)

        L, step_size, data, X = self._setup_11(max_frames, coarseness)

        fig = plt.figure()
        ax = fig.gca()
        ax.plot(X,data[0])

        ani = animation.FuncAnimation(fig, update, len(data), interval=20)

        plt.show()

    def _animate_21(self, max_frames, coarseness):
        """
        Animate the function represented by this network over its training life,
        assuming inputs are 2-dimensional and outputs are 1-dimensional

        Parameters:
            max_frames (integer): The maximum number of frames to use for animations
            coarseness (float): The distance between sample points
        """
        def update(num):
            ax.cla()
            set_axes()
            surf = ax.plot_surface(X,Y,data[num])

        def set_axes():
            ax.set_zlim(0.0, 1.0)

        _, __, data, X, Y = self._setup_21(max_frames, coarseness)

        fig = plt.figure()
        ax = p3.Axes3D(fig)

        surf = ax.plot_surface(X,Y,data[0])
        set_axes()

        ani = animation.FuncAnimation(fig, update, len(data), interval=20)

        plt.show()

    def _animate_12(self, max_frames, coarseness):
        """
        Animate the function represented by this network over its training life,
        assuming inputs are 1-dimensional and outputs are 2-dimensional

        Parameters:
            max_frames (integer): The maximum number of frames to use for animations
            coarseness (float): The distance between sample points
        """
        def update(num):
            ax.cla()
            set_axes()
            line = ax.plot(X,Ys[num],Zs[num])

        def set_axes():
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_zlim(0.0, 1.0)

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        L, step_size, X, Ys, Zs = self._setup_12(max_frames, coarseness)

        line = ax.plot(X,Ys[0],Zs[0])
        set_axes()

        ani = animation.FuncAnimation(fig, update, len(Ys), interval=20)

        plt.show()

    def animate(self, max_frames=200, coarseness=None):
        """
        Animate (if possible), the function represented by this network over its
        training life.
        """
        self._check_dimensions()
        a,b = self._get_dimensions()
        if coarseness == None:
            if (a,b) in ((1,1), (1,2)):
                coarseness = 0.005
            else:
                coarseness = 0.05
        f = getattr(self, '_animate_%s%s' % self._get_dimensions())
        f(max_frames, coarseness)






    def _slider_11(self, max_frames, coarseness):
        """
        Plot the function represented by this network with a slider running over its
        training life, assuming inputs and outputs are both 1-dimensional.

        Parameters:
            max_frames (integer): The maximum number of frames to use for animations
            coarseness (float): The distance between sample points
        """
        def update(num):
            ax.cla()
            set_axes()
            ax.plot(X, data[int(num//step_size)])

        def set_axes():
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)

        L, step_size, data, X = self._setup_11(max_frames, coarseness)

        fig = plt.figure()
        ax = fig.gca()
        ax.plot(X,data[0])
        set_axes()

        axslider = plt.axes([0.03, 0.1, 0.03, 0.65])
        slider = Slider(axslider, 'Time', 0, L-1, valinit=0, valstep=step_size,
                        orientation='vertical')
        slider.on_changed(update)

        plt.show()

    def _slider_21(self, max_frames, coarseness):
        """
        Plot the function represented by this network with a slider running over its
        training life, assuming inputs are 2-dimensional and outputs are 1-dimensional

        Parameters:
            max_frames (integer): The maximum number of frames to use for animations
            coarseness (float): The distance between sample points
        """
        def update(num):
            ax.cla()
            set_axes()
            surf = ax.plot_surface(X,Y,data[int(num//step_size)])

        def set_axes():
            ax.set_zlim(0.0, 1.0)

        L, step_size, data, X, Y = self._setup_21(max_frames, coarseness)

        fig = plt.figure()
        ax = p3.Axes3D(fig)

        surf = ax.plot_surface(X,Y,data[0])
        set_axes()

        axslider = plt.axes([0.03, 0.1, 0.03, 0.65])
        slider = Slider(axslider, 'Time', 0, L-1, valinit=0, valstep=step_size,
                        orientation='vertical')
        slider.on_changed(update)

        plt.show()

    def _slider_12(self, max_frames, coarseness):
        """
        Plot the function represented by this network with a slider running over its
        training life, assuming inputs are 1-dimensional and outputs are 2-dimensional

        Parameters:
            max_frames (integer): The maximum number of frames to use for animations
            coarseness (float): The distance between sample points
        """
        def update(num):
            ax.cla()
            set_axes()
            line = ax.plot(X,Ys[int(num//step_size)],Zs[int(num//step_size)])

        def set_axes():
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ax.set_zlim(0.0, 1.0)

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        L, step_size, X, Ys, Zs = self._setup_12(max_frames, coarseness)

        line = ax.plot(X,Ys[0],Zs[0])
        set_axes()

        axslider = plt.axes([0.03, 0.1, 0.03, 0.65])
        slider = Slider(axslider, 'Time', 0, L-1, valinit=0, valstep=step_size,
                        orientation='vertical')
        slider.on_changed(update)

        plt.show()

    def slider(self, max_frames=200, coarseness=None):
        """
        Plot (if possible), the function represented by this network with a slider
        running over its training life.
        """
        self._check_dimensions()
        a,b = self._get_dimensions()
        if coarseness == None:
            if (a,b) in ((1,1), (1,2)):
                coarseness = 0.01
            else:
                coarseness = 0.05
        f = getattr(self, '_slider_%s%s' % self._get_dimensions())
        f(max_frames, coarseness)

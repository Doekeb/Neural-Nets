import sys
import numpy as np
from scipy.signal import savgol_filter
import networkx as nx
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import itertools as iter
rd = np.random
from matplotlib.widgets import Slider
from matplotlib import animation
import activators as acts
import loss

def sigmoid(z):
    return 1/(1+np.exp(-z))

class Neuron:
    """
    This is a class for a single neuron to be used in a neural network.

    Attributes:
        weights (array of floats): The weights for this neuron's inputs
        bias (float): This neuron's bias
        activator_name (string): The name of this neuron's activator function
        activator (function): This neuron's activator function
        d_activator (function): The derivative of this neuron's activator
    """
    def __init__(self, weights, bias=0, activator=acts.Sigmoid()):
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
        self.activator = activator

    def __repr__(self):
        """
        Return a string representation of this neuron.
        """
        return "A neuron with weights %s, bias %s, and %s activator" \
               % (self.weights, self.bias, self.activator)

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
        self.output = self.activator.f(self.z)
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
        if self.activator == acts.Sigmoid():
            factor = sum(inputs) * self.output * (1-self.output)
        elif self.activator == acts.Tanh():
            factor = sum(inputs) * (1 - self.output**2)
        else:
            factor = sum(inputs) * self.activator.d(self.z)
        old_weights = np.array(self.weights) # Make a copy

        # update the weights with respect to learning rate
        self.weights -= rate * factor * self.inputs

        # update the bias with respect to learning rate
        self.bias -= rate * factor

        return factor * old_weights




# activator can be a single Activator, e.g. Sigmoid() or a dictionary with keywords
# being neuron types and values the Activator, or a list of Activators
# indicating the activator types per-layer, or a list of lists of Activators
# indicating the activator types per-neuron

class NeuralNetwork:
    """
    This is a class for a network of neurons.

    Attributes:
        n_layers (integer): The number of (non-input) layers in this network
        layer_sizes (array of integers): The number of neurons in each layer
        connections (array of matrices): Adjecency matrices between layers
        loss (loss.Loss): The name of the loss function
        neurons (array of arrays of Neurons): The list of Neurons at each layer,
            initiallized randomly
        log (False or dict): If False, training data is not logged. Otherwise, it is
            a dictionary consisting of training data with keywords 'loss',
            'weights', and 'biases', and values the corresponding data after each
            training step.
    """
    def __init__(self, layer_sizes, connections=None, activators=acts.Sigmoid(),
                 log=True, loss=loss.SumOfSquares()):
        """
        Create a new Neural Network instance.

        Parameters:
            layer_sizes (array of integers): The number of neurons in each layer
            connections (array of matrices): Adjecency matrices between layers (if None,
                all neurons in two adjacent layers are connected)
            activators (Activator | array of Activators | dict | array of arrays of Activators):
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

        if isinstance(activators, acts.Activator):
            activators = [[activators]*n for n in layer_sizes[1:]]
        elif type(activators) is dict:
            activators = [[activators['hidden']]*n for n in layer_sizes[1:-1]] \
                       + [[activators['output']]*layer_sizes[-1]]
        elif isinstance(activators[0], acts.Activator):
            activators = [[a]*n for a, n in zip(activators, layer_sizes)]

        self.loss = loss
        self.neurons = [[Neuron(k[i] * a[i].dist(k[i].sum(), n), activator=a[i])
                         for i in range(m)]
                        for (n,m,k,a) in zip(layer_sizes,
                                              layer_sizes[1:],
                                              self.connections,
                                              activators)]

        if log:
            self.log = {'loss':[],
                        'weights':[],
                        'biases':[]}
        else:
            self.log = False

        self.plot = self.Plot(self)

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
                      activator=self.activator, log=self.log,
                      loss=self.loss)

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
                    loss_sum += self.loss.f(results, targets)
                    d_loss_sums += self.loss.d(results, targets)

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

    class Plot:
        def __init__(self, network):
            self.network = network

            if network.log is False:
                self.log = False
            else:
                self.log = True

            if self.log:
                self.loss = network.log['loss']
                self.weights = network.log['weights']
                self.biases = network.log['biases']

        def _get_dimensions(self):
            """
            Return the dimensions of the input and output vectors for this network.
            """
            return (self.network.layer_sizes[0], self.network.layer_sizes[-1])

        def _check_dimensions(self):
            """
            Check that the input and ouput dimensions are appropriate sizes to plot. Raise
            an error if not.
            """
            if self._get_dimensions() not in ((1,1), (1,2), (2,1)):
                msg = '(Input, Output) dimensions must be one of %s, %s, or %s.' \
                      %((1,1), (2,1), (1,2))
                raise(ValueError(msg))

        def _get_filtered_data(self, n_frames):
            """
            Return filtered data from the training log.

            Parameters:
                n_frames (integer): The number of frames to use for animations

            Returns:
                3-tuple: (time stamps to achieve n_frames frames,
                          weight data filtered to n_frames,
                          bias data filtered to n_frames)
            """
            times = np.round(np.linspace(0, len(self.weights)-1, n_frames))
            weights = [self.weights[int(t)] for t in times]
            biases = [self.biases[int(t)] for t in times]

            return (times, weights, biases)

        def _setup_11(self, x_range, n_frames=None, **kwargs):
            """
            Return plotting data assuming the network represents a 1-d to 1-d function.

            Parameters:
                x_range (x_min, x_max, n_points): The minimum and maximum x-values to plot,
                    and the number of points to use
                n_frames (None or integer): The number of frames to use in animations and
                    sliders. If this argument is supplied (and not None), then also supply
                    the keywork argument ax (axes object).

            Returns:
                tuple: If n_frames is None, a 2-tuple (X,Y). Otherwise, a 3-tuple (X,Y,f).
                    X is the x-values to plot, Y is the (most current) y-values to plot,
                    f is an animation/slider update function.
            """
            X = np.linspace(*x_range)
            Y = np.array([self.network.fire([x])[0] for x in X])

            if n_frames is None:
                return (X, Y)

            else:
                _, weights, biases = self._get_filtered_data(n_frames)
                Ys = [np.array([self.network.fire([x], weights=w, biases=b)[0]
                               for x in X])
                     for w, b in zip(weights, biases)]

                ax = kwargs.pop('ax')
                L = len(self.weights)
                step_size = L / n_frames

                def update(num):
                    ax.cla()
                    ax.plot(X, Ys[int(round(num/step_size))], **kwargs)

                return (X, Ys[0], update)

        def _setup_12(self, x_range, n_frames=None, **kwargs):
            """
            Return plotting data assuming the network represents a 1-d to 1-d function.

            Parameters:
                x_range (x_min, x_max, n_points): The minimum and maximum x-values to plot,
                    and the number of points to use
                n_frames (None or integer): The number of frames to use in animations and
                    sliders. If this argument is supplied (and not None), then also supply
                    the keywork argument ax (axes object).

            Returns:
                tuple: If n_frames is None, a 3-tuple (X,Y,Z). Otherwise, a 4-tuple
                    (X,Y,Z,f). X is the x-values to plot, Y is the (most current) y-values
                    to plot, Z is the (most current) z-values to plot, f is an
                    animation/slider update function.
            """
            X = np.linspace(*x_range)
            results = [self.network.fire([x]) for x in X]
            Y = np.array([result[0] for result in results])
            Z = np.array([result[1] for result in results])

            if n_frames is None:
                return (X, Y, Z)

            else:
                _, weights, biases = self._get_filtered_data(n_frames)
                data = [np.array([self.network.fire([x], weights=ws, biases=bs)
                                  for x in X])
                        for ws, bs in zip(weights, biases)]
                Ys = np.array([[result[0] for result in datum]
                               for datum in data])
                Zs = np.array([[result[1] for result in datum]
                               for datum in data])

                ax = kwargs.pop('ax')
                L = len(self.weights)
                step_size = L / n_frames

                def update(num):
                    ax.cla()
                    ax.plot(X, Ys[int(round(num/step_size))],
                            Zs[int(round(num/step_size))], **kwargs)

                return (X, Ys[0], Zs[0], update)

        def _setup_21(self, x_range, y_range, n_frames=None, **kwargs):
            """
            Prepare animation and slider data for 2-d to 1-d functions.

            Parameters:
                x_range (x_min, x_max, n_points): The minimum and maximum x-values to plot,
                    and the number of points to use
                y_range (x_min, x_max, n_points): The minimum and maximum x-values to plot,
                    and the number of points to use
                n_frames (None or integer): The number of frames to use in animations and
                    sliders. If this argument is supplied (and not None), then also supply
                    the keywork argument ax (axes object).

            Returns:
                tuple: If n_frames is None, a 3-tuple (X,Y,Z). Otherwise, a 4-tuple
                    (X,Y,Z,f). X is the x-values to plot, Y is the (most current) y-values
                    to plot, Z is the (most current) z-values to plot, f is an
                    animation/slider update function.
            """
            X = np.linspace(*x_range)
            Y = np.linspace(*y_range)
            Z = np.array([[self.network.fire([x,y])[0] for x in X] for y in Y])

            if n_frames is None:
                return tuple(np.meshgrid(X, Y)) + (Z,)

            else:
                _, weights, biases = self._get_filtered_data(n_frames)
                Zs = [np.array([[self.network.fire([x,y], weights=ws,
                                                   biases=bs)[0]
                                 for x in X] for y in Y])
                      for ws, bs in zip(weights, biases)]
                X, Y = np.meshgrid(X, Y)

                ax = kwargs.pop('ax')
                L = len(self.weights)
                step_size = L / n_frames

                def update(num):
                    ax.cla()
                    ax.plot_surface(X, Y, Zs[int(round(num/step_size))],
                                    **kwargs)

                return (X, Y, Zs[0], update)

        def _setup_graph(self, ax, n_frames=None):
            neurons = self.network.neurons
            n_layers = self.network.n_layers + 1
            layer_sizes = self.network.layer_sizes
            height = max(layer_sizes)
            width = n_layers
            shapes = ['<', 'v', '^', 'd', '8', 'h', 'p', 's', 'o']
            activator_shapes = {'input': '>'}

            G = nx.DiGraph()
            pos = {}
            sh = {}

            # Add all nodes to the graph and set static properties
            for i in range(n_layers):
                n_nodes = layer_sizes[i]
                for n in range(n_nodes):
                    node = (i, n)
                    G.add_node(node)

                    # Determine the position and log it
                    position = np.array([width / n_layers * i,
                                         height - height / (n_nodes+1) * (n+1)])
                    pos[node] = position

                    # Determine the shape to use and log it
                    if i == 0:
                        activator = 'input'
                    else:
                        activator = neurons[i-1][n].activator
                    try:
                        shape = activator_shapes[activator]
                    except KeyError:
                        try:
                            shape = shapes.pop()
                        except IndexError:
                            shape = '<'
                        finally:
                            activator_shapes[activator] = shape
                    sh[node] = shape

            # Add all edges to the graph
            for i in range(n_layers-1):
                for n in range(layer_sizes[i]):
                    for m in range(layer_sizes[i+1]):
                        edge = ((i,n), (i+1,m))
                        G.add_edges_from([edge])

            def draw(weights, biases):
                for node in G.nodes():
                    i, n = node
                    if i == 0:
                        color = (0.0, 0.0, 0.0, 1.0)
                    else:
                        bias = biases[i-1][n]
                        # color = (1-sigmoid(bias), 0.0, sigmoid(bias),
                        #          abs(2*(sigmoid(bias)-1/2)))
                        color = (1-sigmoid(bias), 0.0, sigmoid(bias), 1.0)
                    nx.draw_networkx_nodes(G,
                                           pos=pos,
                                           nodelist=[node],
                                           node_color=[color],
                                           edgecolors=[(0.0, 0.0, 0.0, 1.0)],
                                           node_shape=sh[node],
                                           ax=ax)
                max_thickness = 20
                for edge in G.edges:
                    s, t = edge
                    i, n = s
                    _, m = t
                    weight = weights[i][m][n]
                    alpha = abs(2*(sigmoid(weight)-1/2))
                    color = (1-sigmoid(weight), 0.0, sigmoid(weight), alpha)
                    width = max_thickness * alpha
                    nx.draw_networkx_edges(G,
                                           pos=pos,
                                           edgelist=[edge],
                                           width=width,
                                           edge_color=color,
                                           arrows=False,
                                           ax=ax)

            if n_frames is None:
                draw(self.network.get_weights(), self.network.get_biases())

            else:
                _, weights, biases = self._get_filtered_data(n_frames)
                L = len(self.weights)
                step_size = L / n_frames

                draw(weights[0], biases[0])

                def update(num):
                    ax.cla()
                    draw(weights[int(round(num/step_size))],
                         biases[int(round(num/step_size))])

                return update

        def _setup_loss(self, n_frames=None, ax=None):
            L = len(self.loss)
            data = savgol_filter(self.loss, 2*L//100+1, 3)
            minimum = min(data)
            maximum = max(data)
            if n_frames is None:
                return data
            else:
                step_size = L / n_frames
                def update(num):
                    ax.cla()
                    set_axes()
                    ax.plot(data[:int(round(num))])

                def set_axes():
                    ax.set_xlim(0, L)
                    ax.set_ylim(minimum, maximum)

                return data, update, set_axes

        def _plot_function_11(self, x_range, **kwargs):
            """
            Plot the function represented by this network, assuming inputs and outputs are
            both one-dimensional.

            Parameters:
                x_range (x_min, x_max, n_points): The minimum and maximum x-values to plot,
                    and the number of points to use
            """
            X, Y = self._setup_11(x_range, **kwargs)
            plt.plot(X, Y, **kwargs)

        def _plot_function_12(self, x_range, **kwargs):
            """
            Plot the function represented by this network, assuming inputs are 1-dimensional
            and outputs are 2-dimensional.

            Parameters:
                x_range (x_min, x_max, n_points): The minimum and maximum x-values to plot,
                    and the number of points to use
            """
            X, Y, Z = self._setup_12(x_range, **kwargs)

            ax = plt.gca(projection='3d')
            ax.plot(X, Y, Z, **kwargs)

        def _plot_function_21(self, input_range, **kwargs):
            """
            Plot the function represented by this network, assuming inputs are 2-dimensional
            and outputs are 1-dimensional.

            Parameters:
                input_range (tuple): A pair of 3-tuples, each consisting of minimum input,
                    maximum input, and the number of input values for the two input axes.
            """
            X, Y, Z = self._setup_21(input_range[0], input_range[1], **kwargs)

            ax = plt.gca(projection='3d')
            ax.plot_surface(X,Y,Z)

        def plot_function(self, input_range, **kwargs):
            """
            Plot (if possible) the function represented by this network.

            Parameters:                                                                     ;;
                input_range (tuple): If inputs to the network are 1-d, a 3-tuple consisting
                    of minimum input, maximum input, and the number of input values. If
                    inputs to the network are 2-d, a pair of 3-tuples, each consisting of
                    the above data for the two input axes.
                kwargs (keyword arguments): Will be passed to pyplot commands (for example,
                    output range data can be passed here).
            """
            self._check_dimensions()
            a,b = self._get_dimensions()
            f = getattr(self, '_plot_function_%s%s' % (a,b))
            f(input_range, **kwargs)

        def plot_graph(self):
            ax = plt.gca()
            self._setup_graph(ax=ax)

        def plot_loss(self):
            """
            Plot the (smoothed) loss function over this network's training life
            """
            data = self._setup_loss()
            plt.plot(data)

        def plot_all(self, input_range):
            self._check_dimensions()
            a,b = self._get_dimensions()
            f = getattr(self, '_plot_function_%s%s' % (a,b))

            plt.figure(figsize=(20,5))
            plt.subplot(131)
            f(input_range)
            plt.subplot(132)
            self.plot_graph()
            plt.subplot(133)
            self.plot_loss()

        def _animate_function_11(self, x_range, n_frames, interval, **kwargs):
            """
            Animate the function represented by this network, assuming inputs and ouputs are
            both 1-d

            Parameters:
                x_range (x_min, x_max, n_points): The minimum and maximum x-values to plot,
                    and the number of points to use
                n_frames (integer): The number of frames in the animation
                kwargs (keyword arguments): Will be passed to pyplot commands (for example,
                    output range data can be passed here).
            """
            ax = plt.gca()
            fig = ax.get_figure()

            X, Y, update = self._setup_11(x_range, n_frames=n_frames, ax=ax,
                                          **kwargs)
            L = len(self.weights)

            ax.plot(X, Y, **kwargs)

            return animation.FuncAnimation(fig, update,
                                           np.linspace(0, L, n_frames,
                                                       endpoint=False),
                                           interval=interval)

        def _animate_function_12(self, x_range, n_frames, interval, **kwargs):
            """
            Animate the function represented by this network over its training life,
            assuming inputs are 1-dimensional and outputs are 2-dimensional

            Parameters:
                x_range (x_min, x_max, n_points): The minimum and maximum x-values to plot,
                    and the number of points to use
                n_frames (integer): The number of frames in the animation
                kwargs (keyword arguments): Will be passed to pyplot commands (for example,
                    output range data can be passed here).
            """
            ax = plt.gca(projection='3d')
            fig = ax.get_figure()

            X, Y, Z, update = self._setup_12(x_range, n_frames=n_frames, ax=ax,
                                             **kwargs)
            L = len(self.weights)

            ax.plot(X, Y, Z, **kwargs)

            return animation.FuncAnimation(fig, update,
                                           np.linspace(0, L, n_frames,
                                                       endpoint=False),
                                           interval=interval)

        def _animate_function_21(self, input_range, n_frames, interval, **kwargs):
            """
            Animate the function represented by this network over its training life,
            assuming inputs are 2-dimensional and outputs are 1-dimensional

            Parameters:
                input_range (tuple): A pair of 3-tuples, each consisting of minimum input,
                    maximum input, and the number of input values for the two input axes.
                n_frames (integer): The number of frames in the animation
                kwargs (keyword arguments): Will be passed to pyplot commands (for example,
                    output range data can be passed here).
            """

            ax = plt.gca(projection='3d')
            fig = ax.get_figure()

            X, Y, Z, update = self._setup_21(input_range[0], input_range[1],
                                             n_frames, ax=ax, **kwargs)
            L = len(self.weights)

            ax.plot_surface(X, Y, Z, **kwargs)

            return animation.FuncAnimation(fig, update,
                                           np.linspace(0, L, n_frames,
                                                       endpoint=False),
                                           interval=interval)

        def animate_function(self, input_range, n_frames=200, interval=20, **kwargs):
            """
            Animate (if possible) the function represented by this network.

            Parameters:
                input_range (tuple): If inputs to the network are 1-d, a 3-tuple consisting
                    of minimum input, maximum input, and the number of input values. If
                    inputs to the network are 2-d, a pair of 3-tuples, each consisting of
                    the above data for the two input axes.
                n_frames (integer): The number of frames in the animation
                kwargs (keyword arguments): Will be passed to pyplot commands (for example,
                    output range data can be passed here).
            """
            self._check_dimensions()
            a,b = self._get_dimensions()
            f = getattr(self, '_animate_function_%s%s' % (a,b))
            return f(input_range, n_frames, interval, **kwargs)

        def animate_graph(self, n_frames=200, interval=50):
            ax = plt.gca()
            fig = ax.get_figure()
            update = self._setup_graph(ax, n_frames)

            L = len(self.weights)

            return animation.FuncAnimation(fig, update,
                                           np.linspace(0, L, n_frames,
                                                       endpoint=False),
                                           interval=interval)

        def animate_loss(self, n_frames=200, interval=50):
            ax = plt.gca()
            fig = ax.get_figure()

            data, update, set_axes = self._setup_loss(n_frames=n_frames, ax=ax)

            L = len(self.loss)

            ax.plot(data[:1])
            set_axes()

            return animation.FuncAnimation(fig, update,
                                           np.linspace(0, L, n_frames,
                                                       endpoint=False),
                                           interval=interval)

        def _animate_all_11(self, x_range, n_frames, interval):
            L = len(self.loss)
            dims = plt.figaspect(0.3)
            fig = plt.figure(figsize=dims)

            ax_function = plt.subplot(131)
            X, Y, update_function = self._setup_11(x_range, n_frames=n_frames,
                                                   ax=ax_function)
            ax_function.plot(X, Y)

            ax_graph = plt.subplot(132)
            update_graph = self._setup_graph(ax=ax_graph, n_frames=n_frames)

            ax_loss = plt.subplot(133)
            data, update_loss, set_axes = self._setup_loss(ax=ax_loss, n_frames=n_frames)
            ax_loss.plot(data[:1])
            set_axes()

            def update(num):
                for f in [update_function, update_graph, update_loss]:
                    f(num)

            return animation.FuncAnimation(fig, update,
                                           np.linspace(0, L, n_frames,
                                                       endpoint=False),
                                           interval=interval)

        def _animate_all_12(self, x_range, n_frames, interval):
            pass

        def _animate_all_21(self, input_range, n_frames, interval):
            pass

        def animate_all(self, input_range, n_frames=100, interval=100):
            self._check_dimensions()
            a,b = self._get_dimensions()
            f = getattr(self, '_animate_all_%s%s' % (a,b))
            return f(input_range, n_frames, interval)

        def _slider_function_11(self, x_range, n_frames, **kwargs):
            """
            Plot the function represented by this network with a slider running over its
            training life, assuming inputs and outputs are both 1-dimensional.

            Parameters:
                x_range (x_min, x_max, n_points): The minimum and maximum x-values to plot,
                    and the number of points to use
                n_frames (integer): The number of frames in the animation
                kwargs (keyword arguments): Will be passed to pyplot commands (for example,
                    output range data can be passed here).
            """
            _, (ax, axslider) = plt.subplots(2, 1, gridspec_kw={'height_ratios':(10,1)})

            X, Y, update = self._setup_11(x_range, n_frames=n_frames, ax=ax,
                                          **kwargs)

            ax.plot(X, Y, **kwargs)

            L = len(self.weights)
            step_size = L / n_frames

            slider = Slider(axslider, 'Time', 0, L-step_size, valinit=0,
                            valstep=step_size, orientation='horizontal')
            slider.on_changed(update)

            return slider

        def _slider_function_12(self, x_range, n_frames, **kwargs):
            """
            Plot the function represented by this network with a slider running over its
            training life, assuming inputs are 1-dimensional and outputs are 2-dimensional

            Parameters:
                x_range (x_min, x_max, n_points): The minimum and maximum x-values to plot,
                    and the number of points to use
                n_frames (integer): The number of frames in the animation
                kwargs (keyword arguments): Will be passed to pyplot commands (for example,
                    output range data can be passed here).
            """
            ax = plt.gca(projection='3d')

            X, Y, Z, update = self._setup_12(x_range, n_frames=n_frames, ax=ax,
                                             **kwargs)

            ax.plot(X, Y, Z, **kwargs)

            L = len(self.weights)
            step_size = L / n_frames

            axslider = plt.axes([0.03, 0.1, 0.03, 0.65])
            slider = Slider(axslider, 'Time', 0, L-step_size, valinit=0,
                            valstep=step_size, orientation='vertical')
            slider.on_changed(update)

            return slider

        def _slider_function_21(self, input_range, n_frames, **kwargs):
            """
            Plot the function represented by this network with a slider running over its
            training life, assuming inputs are 2-dimensional and outputs are 1-dimensional

            Parameters:
                input_range (tuple): A pair of 3-tuples, each consisting of minimum input,
                    maximum input, and the number of input values for the two input axes.
                n_frames (integer): The number of frames in the animation
                kwargs (keyword arguments): Will be passed to pyplot commands (for example,
                    output range data can be passed here).
            """
            ax = plt.gca(projection='3d')

            X, Y, Z, update = self._setup_21(input_range[0], input_range[1],
                                             n_frames, ax=ax, **kwargs)

            ax.plot_surface(X, Y, Z, **kwargs)

            L = len(self.weights)
            step_size = L / n_frames

            axslider = plt.axes([0.03, 0.1, 0.03, 0.65])
            slider = Slider(axslider, 'Time', 0, L-step_size, valinit=0,
                            valstep=step_size, orientation='vertical')
            slider.on_changed(update)

            return slider

        def slider_function(self, input_range, n_frames=200, **kwargs):
            """
            Parameters:
                input_range (tuple): If inputs to the network are 1-d, a 3-tuple consisting
                    of minimum input, maximum input, and the number of input values. If
                    inputs to the network are 2-d, a pair of 3-tuples, each consisting of
                    the above data for the two input axes.
                n_frames (integer): The number of frames in the animation
                kwargs (keyword arguments): Will be passed to pyplot commands (for example,
                    output range data can be passed here).
            """
            self._check_dimensions()
            a,b = self._get_dimensions()
            f = getattr(self, '_slider_function_%s%s' % (a,b))
            return f(input_range, n_frames, **kwargs)

        def slider_graph(self, n_frames=200):
            ax = plt.gca()
            update = self._setup_graph(ax, n_frames)

            L = len(self.weights)
            step_size = L / n_frames

            axslider = plt.axes([0.03, 0.1, 0.03, 0.65])
            slider = Slider(axslider, 'Time', 0, L-step_size, valinit=0,
                            valstep=step_size, orientation='vertical')
            slider.on_changed(update)

            return slider

        def slider_loss(self, n_frames=200):
            ax = plt.gca()
            data, update, set_axes = self._setup_loss(n_frames=n_frames, ax=ax)

            ax.plot(data[:1])
            set_axes()

            L = len(self.loss)
            step_size = L / n_frames

            axslider = plt.axes([0.03, 0.1, 0.03, 0.65])
            slider = Slider(axslider, 'Time', 0, L-step_size, valinit=0,
                            valstep=step_size, orientation='vertical')
            slider.on_changed(update)

            return slider

        def _slider_all_11(self, x_range, n_frames):
            L = len(self.loss)
            step_size = L / n_frames
            dims = plt.figaspect(0.3)

            _, ((ax_function, ax_graph, ax_loss), (dump1, axslider, dump2)) = plt.subplots(2, 3, gridspec_kw={'height_ratios':(10,1)}, figsize=dims)
            dump1.remove(), dump2.remove()

            X, Y, update_function = self._setup_11(x_range, n_frames=n_frames,
                                                   ax=ax_function)
            update_graph = self._setup_graph(ax=ax_graph, n_frames=n_frames)
            data, update_loss, set_axes = self._setup_loss(ax=ax_loss, n_frames=n_frames)

            ax_function.plot(X, Y)
            ax_loss.plot(data[:1])
            set_axes()

            def update(num):
                for f in [update_function, update_graph, update_loss]:
                    f(num)

            slider = Slider(axslider, 'Time', 0, L-step_size, valinit=0,
                            valstep=step_size, orientation='horizontal')
            slider.on_changed(update)

            return slider

        def _slider_all_12(self, x_range, n_frames):
            pass

        def _slider_all_21(self, input_range, n_frames):
            pass

        def slider_all(self, input_range, n_frames=200):
            self._check_dimensions()
            a,b = self._get_dimensions()
            f = getattr(self, '_slider_all_%s%s' % (a,b))
            return f(input_range, n_frames)

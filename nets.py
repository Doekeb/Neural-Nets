import sys
import numpy as np
from matplotlib import pyplot as plt
import itertools as iter
rd = np.random
import plotting
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.widgets import Slider
from copy import deepcopy
from matplotlib import animation

def sigmoid(z): # sigmoidal activator function (between 0 and 1)
    return 1/(1+np.exp(-z))

def d_sigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))

def arctan(z): # (shifted and scaled) arctan activator function (between 0 and 1)
    return np.arctan(z)/np.pi + np.float64(1)/2

def d_arctan(z):
    return 1/(np.pi*(1+z**2))

def exponential(z): # piecewise exponential activator function (between 0 and 1)
    if z <= 0:
        return 1/2*np.exp(z)
    if z >= 0:
        return 1-1/2*np.exp(-z)

def d_exponential(z):
    if z <= 0:
        return 1/2*np.exp(z)
    if z >= 0:
        return 1/2*np.exp(-z)

def tanh(z): # (shifted) hyperbolic tangent activator function
    return 1/2*(np.tanh(z)+1)

def d_tanh(z):
    return 1/2*(1 - np.tanh(z)**2)

def quadratic(z): # piecewise quadratic activator function (bad choice)
    if z <= -1 or z >= 1:
        return np.float64(0)
    if z >= -1 and z <= 0:
        return np.float64(1)/2*(z+1)**2
    if z >=0 and z <= 1:
        return -np.float64(1)/2*(z-1)**2+1

def d_quadratic(z):
    if z <= -1 or z >= 1:
        return np.float64(0)
    if z >= -1 and z <= 0:
        return z+np.float64(1)
    if z >=0 and z <= 1:
        return np.float64(1)-z

class Neuron():
    def __init__(self, weights, bias, activator='sigmoid'):
        self.weights = np.array(weights, np.float64)
        self.bias = np.float64(bias)
        self.activator_name = activator

        # Get the activator function by name
        self.activator = getattr(sys.modules[__name__], activator)
        self.d_activator = getattr(sys.modules[__name__], 'd_' + activator)

    def __repr__(self):
        return "A neuron with weights %s, bias %s, and %s activator" % (self.weights, self.bias, self.activator_name)

    def fire(self, inputs, weights=None, bias=None):
        if weights is None:
            weights = self.weights
        if bias is None:
            bias = self.bias
        self.inputs = np.array(inputs, np.float64)
        self.z = np.dot(inputs, weights) + bias
        self.output = self.activator(self.z)
        return self.output

    def back_prop(self, inputs, rate=0.1):
        # sigmoid and tanh have nice derivatives
        if self.activator_name == 'sigmoid':
            factor = sum(inputs) * self.output * (1-self.output)
        elif self.activator_name == 'tanh':
            factor = sum(inputs) * (1-self.output**2)
        else:
            factor = sum(inputs) * self.d_activator(self.z)
        old_weights = np.array(self.weights) # Make a copy

        # update the weights with respect to learning rate
        self.weights -= rate * factor * self.inputs

        # update the bias with respect to learning rate
        self.bias -= rate * factor

        return factor * old_weights

test_set = {(1,1):1, (1,0):1, (0,1):1, (0,0):0}
n_samples = 1000
training_set = {(x,y):(np.round(x) or np.round(y)) for (x,y) in [rd.rand(2) for _ in range(n_samples)]}

def sum_of_squares(results, targets):
    return sum((results-targets)**2)

def d_sum_of_squares(results, targets):
    return 2*(results-targets)

def loss(result, target):
    return (result-target)**2

def d_loss(result, target):
    return 2*(result-target)

class LogicGateNetwork():
    def __init__(self, activator='sigmoid', log=True, loss='sum_of_squares'):
        self.activator_name = activator
        self.activator = getattr(sys.modules[__name__], activator)
        self.loss_name = loss
        self.loss = getattr(sys.modules[__name__], loss)
        self.d_loss = getattr(sys.modules[__name__], 'd_'+loss)
        self.neuron = Neuron(rd.normal(size=2), rd.normal(), activator=activator)
        self.log = log and {'loss':[],
                            'weights':[(self.neuron.weights[0], self.neuron.weights[1])],
                            'bias':[self.neuron.bias]}

    def reset(self):
        if self.log:
            self.log = True
        self.__init__(activator=self.activator_name, log=self.log, loss=self.loss_name)

    def train(self, data, rate=0.1, batch_size=1, epochs=1):
        for _ in range(epochs): # Repeat for how many epochs there are

            # Get the data and randomize it
            items = list(data.items())
            rd.shuffle(items)

            # Form the batches
            n_leftovers = len(data) % batch_size
            if n_leftovers == 0:
                batches = [items[batch_size*i:batch_size*(i+1)] for i in range(len(items) // batch_size)]
            else:
                batches = [items[batch_size*i:batch_size*(i+1)] for i in range(len(items) // batch_size)] \
                        + [items[-n_leftovers:]]

            # Adjust weights once for each batch
            for batch in batches:
                loss_sum = 0
                d_loss_sum = 0

                # Fire the network once for each sample
                for (key, target) in batch:
                    result = self.neuron.fire(np.array(key, np.float64))
                    loss_sum += self.loss([result], [target])
                    d_loss_sum += self.d_loss([result], [target])[0]

                # Average the loss and its derivative
                loss_average = loss_sum / len(batch)
                d_loss_average = d_loss_sum / len(batch)

                # Back-propogate with loss average
                self.neuron.back_prop(np.array([d_loss_average]), rate=rate)

                if self.log: # Log result
                    self.log['loss'] += [loss_average]
                    self.log['weights'] += [(self.neuron.weights[0],self.neuron.weights[1])]
                    self.log['bias'] += [self.neuron.bias]

    # Fire the network with the given inputs
    def fire(self, inputs):
        return self.neuron.fire(inputs)

    # Test the network on the (labelled) data set
    def test(self, data):
        for (key, value) in data.items():
            result = self.neuron.fire(np.array(key, np.float64))
            print("Input: %s, Output: %s, Expected: %s" % (key, result, value))

    # Plot the (smoothed) loss function over this network's training life
    def plot_loss(self):
        log = np.array(self.log['loss'])
        L = len(log)
        plotting.plot_moving_average(log, window=int(np.ceil(L/100)))

    # Animate the function represented by this network
    def animate(self, max_frames=200):
        # This is more an order of magnitude than an actual max size
        L = len(self.log['weights'])
        if L > max_frames:
            weights = self.log['weights'][::L//max_frames]
            bias = self.log['bias'][::L//max_frames]
        else:
            weights = self.log['weights']
            bias = self.log['bias']
        functions = [(lambda z: (lambda x, y: self.activator(z[0]*x + z[1]*y + z[2])))((w0, w1, b)) for ((w0, w1), b) in zip(weights, bias)]
        plotting.animate(functions)

'''
test_set = {(i,j,k): np.array([i or j, k]) for i, j, k in iter.product([0,1], repeat=3)}
n_samples = 1000
training_set = {(x,y):(np.round(x) or np.round(y)) for (x,y) in [rd.rand(2) for _ in range(n_samples)]}
'''

test_set = {(i,j): np.array([i or j]) for i,j in iter.product([0,1], repeat=2)}
n_samples = 1000
training_set = {(x,y): np.array([np.round(x) or np.round(y)]) for (x,y) in [rd.rand(2) for _ in range(n_samples)]}


# TODO: Eliminate the connections matrix. It's unnecessary.

class NeuralNetwork():
    def __init__(self, layer_sizes, connections=None, activator='sigmoid', log=True, loss='sum_of_squares'):
        self.n_layers = len(layer_sizes) - 1
        self.layer_sizes = layer_sizes
        if connections == None:
            self.connections = [np.ones((m,n)) for (n,m) in zip(layer_sizes, layer_sizes[1:])]
        else:
            self.connections = [np.array(connection) for connection in connections]
        self.activator_name = activator
        self.activator = getattr(sys.modules[__name__], activator)
        self.loss_name = loss
        self.loss = getattr(sys.modules[__name__], loss)
        self.d_loss = getattr(sys.modules[__name__], 'd_'+loss)
        self.neurons = [[Neuron(k[i] * rd.normal(size=n), rd.normal(), activator=activator)
                         for i in range(m)]
                         for (n,m,k) in zip(layer_sizes, layer_sizes[1:], self.connections)]
        self.log = log and {'loss':[],
                            'weights':[self.get_weights()],
                            'biases':[self.get_biases()]}

    def get_weights(self):
        return [self.connections[i] * np.array([N.weights for N in self.neurons[i]]) for i in range(self.n_layers)]

    def get_biases(self):
        return [np.array([N.bias for N in self.neurons[i]]) for i in range(self.n_layers)]

    def reset(self):
        if self.log:
            self.log = True
        self.__init__(self.layer_sizes, connections=self.connections, activator=self.activator_name, log=self.log, loss=self.loss_name)

    # Fire the network with the given inputs
    def fire(self, inputs, weights=None, biases=None):
        results = np.array(inputs)
        if weights is None and biases is None:
            for c_matrix, n_list in zip(self.connections, self.neurons):
                results = np.array([N.fire(row*results) for (row, N) in zip(c_matrix, n_list)])
        else:
            for c_matrix, n_list, w_matrix, b_array in zip(self.connections, self.neurons, weights, biases):
                results = np.array([N.fire(row*results, weights=w, bias=b) for (row, N, w, b) in zip(c_matrix, n_list, w_matrix, b_array)])
        return np.array(results)

    def back_prop(self, inputs, rate):
        results = np.diag(inputs)
        for n_list in reversed(self.neurons):
            results = np.array([N.back_prop(result) for (N, result) in zip(n_list, results.T)])

    # Test the network on the (labelled) data set
    def test(self, data):
        for (key, value) in data.items():
            result = self.fire(np.array(key, np.float64))
            print("Input: %s, Output: %s, Expected: %s" % (key, result, value))

    def train(self, data, rate=0.1, batch_size=1, epochs=1):
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
                    results = self.fire(np.array(key, np.float64))
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

    # Plot the (smoothed) loss function over this network's training life
    def plot_loss(self):
        log = np.array(self.log['loss'])
        L = len(log)
        plotting.plot_moving_average(log, window=int(np.ceil(L/100)))

    def plot_function(self):
        fig = plt.figure()
        ax = p3.Axes3D(fig)
        ax.set_zlim(0.0, 1.0)

        X = np.arange(0, 1.05, 0.05)
        Y = np.arange(0, 1.05, 0.05)
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


    # TODO: FIX THIS
    # Only try this for inpuit size == 2 and output size == 1
    def animate(self, max_frames=200):
        L = len(self.log['weights'])
        if L > max_frames:
            weights = np.array(self.log['weights'][::L//max_frames])
            biases = np.array(self.log['biases'][::L//max_frames])
        else:
            weights = np.array(self.log['weights'])
            biases = np.array(self.log['biases'])

        X = np.arange(0, 1.05, 0.05)
        Y = np.arange(0, 1.05, 0.05)
        data = np.array([[[self.fire([x,y], weights=ws, biases=bs)[0] for x in X] for y in Y] for ws, bs in zip(weights, biases)])
        X, Y = np.meshgrid(X,Y)

        fig = plt.figure()
        ax = p3.Axes3D(fig)

        def update(num, data, surf):
            ax.cla()
            set_axes()
            surf = ax.plot_surface(X,Y,data[num])

        def set_axes():
            ax.set_zlim(0.0, 1.0)

        surf = ax.plot_surface(X,Y,data[0])
        set_axes()

        ani = animation.FuncAnimation(fig, update, len(data), fargs=(data,surf), interval=20)

        plt.show()

    def slider(self, max_frames=200):
        L = len(self.log['weights'])
        if L > max_frames:
            weights = np.array(self.log['weights'][::L//max_frames])
            biases = np.array(self.log['biases'][::L//max_frames])
        else:
            weights = np.array(self.log['weights'])
            biases = np.array(self.log['biases'])
        L = len(weights)

        X = np.arange(0, 1.05, 0.05)
        Y = np.arange(0, 1.05, 0.05)
        data = np.array([[[self.fire([x,y], weights=ws, biases=bs)[0] for x in X] for y in Y] for ws, bs in zip(weights, biases)])
        X, Y = np.meshgrid(X,Y)

        fig = plt.figure()
        ax = p3.Axes3D(fig)

        def update(num):
            ax.cla()
            set_axes()
            surf = ax.plot_surface(X,Y,data[int(num)])

        def set_axes():
            ax.set_zlim(0.0, 1.0)

        surf = ax.plot_surface(X,Y,data[0])
        set_axes()

        axslider = plt.axes([0.03, 0.1, 0.03, 0.65])
        slider = Slider(axslider, 'Time', 0, L-1, valinit=0, valstep=1, orientation='vertical')
        slider.on_changed(update)

        # ani = animation.FuncAnimation(fig, update, len(data), fargs=(data,surf), interval=20)

        plt.show()

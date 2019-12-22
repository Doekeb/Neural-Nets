import sys
import numpy as np
from matplotlib import pyplot as plt
rd = np.random
import plotting

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

    def fire(self, inputs):
        self.inputs = np.array(inputs, np.float64)
        self.z = np.dot(inputs, self.weights)+self.bias
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
    return sum([(result-target)**2 for result, target in zip(results, targets)])

def d_sum_of_squares(results, targets, index):
    return sum([2*(results[index]-targets[index])])

def loss(result, target):
    return (result-target)**2

def d_loss(result, target):
    return 2*(result-target)

class LogicGateNetwork():
    def __init__(self, activator='sigmoid', log=True, loss='sum_of_squares'):
        self.log = log
        self.activator = getattr(sys.modules[__name__], activator)
        self.loss = getattr(sys.modules[__name__], loss)
        self.d_loss = getattr(sys.modules[__name__], 'd_'+loss)
        self.neuron = Neuron(rd.normal(size=2), rd.normal(), activator=activator)
        if log:
            # note: the array of losses will always be one smaller than the array of weights and biases
            self.log = {'loss':[], 'weights':[(self.neuron.weights[0], self.neuron.weights[1])], 'bias':[self.neuron.bias]}

    def reset(self):
        if self.log:
            self.log = True
        self.__init__(log=self.log)

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
                batches = [items[batch_size*i:batch_size*(i+1)] for i in range(len(items) // batch_size)] + [items[-n_leftovers:]]

            # Adjust weights once for each batch
            for batch in batches:
                loss_sum = 0
                d_loss_sum = 0

                # Fire the network once for each sample
                for (key, target) in batch:
                    result = self.neuron.fire(np.array(key, np.float64))
                    loss_sum += self.loss([result], [target])
                    d_loss_sum += self.d_loss([result], [target], 0)

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

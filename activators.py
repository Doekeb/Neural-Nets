from abc import ABC, abstractmethod
import numpy as np
rd = np.random

class Activator(ABC):
    def __repr__(self):
        return 'A neuron activator'

    @abstractmethod
    def function(self, z):
        pass

    f = function

    @abstractmethod
    def derivative(self, z):
        pass

    d = derivative

    @abstractmethod
    def distribution(self, n):
        pass

    dist = distribution

class Sigmoid(Activator):
    """
    Sigmoidal activator function (between 0 and 1)
    """
    def function(self, z):
        return 1/(1+np.exp(-z))

    def derivative(self, z):
        return sigmoid(z)*(1-sigmoid(z))

    def distribution(self, n):
        return rd.randn()*np.sqrt(1/n)

class Arctan(Activator):
    """
    arc-tangent activator function (between 0 and 1)
    """
    def function(self, z):
        return np.arctan(z)/np.pi + np.float64(1)/2

    def derivative(self, z):
        return 1/(np.pi*(1+z**2))

    def distribution(self, n):
        return rd.randn()*np.sqrt(1/n)

class Exponential(Activator):
    """
    Piecewise exponential activator function (between 0 and 1)
    """
    def function(self, z):
        if z <= 0:
            return 1/2*np.exp(z)
        if z >= 0:
            return 1-1/2*np.exp(-z)

    def derivative(self, z):
        if z <= 0:
            return 1/2*np.exp(z)
        if z >= 0:
            return 1/2*np.exp(-z)

    def distribution(self, n):
        return rd.randn()*np.sqrt(1/n)

class Tanh(Activator):
    """
    Hyperbolic tangent activator function
    """
    def function(self, z):
        return np.tanh(z)

    def derivative(self, z):
        return 1-np.tanh(z)**2

    def distribution(self, n):
        return rd.randn()*np.sqrt(1/n)

class Relu(Activator):
    """
    Rectified linear unit activator function
    """
    def function(self, z):
        if z >= 0:
            return z
        if z <= 0:
            return np.float64(0)

    def derivative(self, z):
        if z >= 0:
            return np.float64(1)
        if z <= 0:
            return np.float64(0)

    def distribution(self, n):
        return rd.randn()*np.sqrt(2/n)

class Lrelu(Activator):
    """
    Leaky rectified linear unit activator function
    """
    def __init__(self, rate=0.01):
        self.rate = rate

    def function(self, z):
        if z >= 0:
            return z
        if z <= 0:
            return self.rate*z

    def derivative(self, z):
        if z >= 0:
            return np.float64(1)
        if z <= 0:
            return self.rate

    def distribution(self, n):
        return rd.randn()*np.sqrt(2/n)

class Linear(Activator):
    """
    Linear activator function
    """
    def function(self, z):
        return z

    def derivative(self, z):
        return np.float64(1)

    def distribution(self, n):
        return rd.randn()*np.sqrt(2/n)

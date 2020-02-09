from abc import ABC, abstractmethod
import numpy as np
rd = np.random

class Activator(ABC):
    def __init__(self):
        # These lines just give nice shorthands for the class methods
        self.f = self.function
        self.d = self.derivative
        self.dist = self.distribution

    def __repr__(self):
        pass

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    @abstractmethod
    def function(self, z):
        pass

    @abstractmethod
    def derivative(self, z):
        pass

    def distribution(self, n, k):
        """
        Return a random array of values from an appropriate initialization
        distribution for this activator. n is the number of inputs this activator
        has, m is the length of the array.
        """
        return np.sqrt(1/n) * np.array([rd.randn() for _ in range(k)])

class Sigmoid(Activator):
    """
    Sigmoidal activator function (between 0 and 1)
    """
    def __repr__(self):
        return "Sigmoid activator"

    def function(self, z):
        return 1/(1+np.exp(-z))

    def derivative(self, z):
        return np.exp(-z) / (np.exp(-z) + 1)**2

class Arctan(Activator):
    """
    arc-tangent activator function
    """
    def __repr__(self):
        return "Arctan activator"

    def function(self, z):
        return np.arctan(z)

    def derivative(self, z):
        return 1/(1+z**2)

class Exponential(Activator):
    """
    Piecewise exponential activator function (between 0 and 1)
    """
    def __repr__(self):
        return "Exponential activator"

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

class Tanh(Activator):
    """
    Hyperbolic tangent activator function
    """
    def __repr__(self):
        return "Hyperbolic tangent activator"

    def function(self, z):
        return np.tanh(z)

    def derivative(self, z):
        return 1-np.tanh(z)**2

class Relu(Activator):
    """
    Rectified linear unit activator function
    """
    def __repr__(self):
        return "Rectified linear activator"

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

    def distribution(self, n, k):
        return np.sqrt(2/n) * np.array([rd.randn() for _ in range(k)])

class Lrelu(Activator):
    """
    Leaky rectified linear unit activator function
    """
    def __repr__(self):
        return "Leaky rectified linear activator"

    def __init__(self, rate=0.01):
        self.rate = rate
        Activator.__init__(self)

    def __eq__(self, other):
        return (type(self) == type(other)) and (self.rate == other.rate)

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

    def distribution(self, n, k):
        return np.sqrt(2/n) * np.array([rd.randn() for _ in range(k)])

class Linear(Activator):
    """
    Linear activator function
    """
    def __repr__(self):
        return "Linear activator"

    def function(self, z):
        return z

    def derivative(self, z):
        return np.float64(1)

    def distribution(self, n, k):
        return np.sqrt(2/n) * np.array([rd.randn() for _ in range(k)])

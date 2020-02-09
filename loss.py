from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):
    def __init__(self):
        # These lines just give nice shorthands for the class methods
        self.f = self.function
        self.d = self.derivative

    def __repr__(self):
        pass

    def __eq__(self, other):
        return type(self) == type(other)

    @abstractmethod
    def function(self, results, target):
        pass

    @abstractmethod
    def derivative(self, results, targets):
        pass

class SumOfSquares(Loss):
    """
    Sum of squares loss function
    """
    def __repr__(self):
        return "Sum of squares loss function"

    def function(self, results, target):
        """
        Parameters:
            results (array of floats): The observed values
            targets (array of floats): The desired values

        Returns:
            float: Measure of error between results and targets
        """
        return sum((results-targets)**2)

    def derivative(self, results, target):
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

class MeanSquaredError(Loss):
    """
    Mean squared error loss function
    """
    def __repr__(self):
        return "Mean squared error loss function"

    def function(self, results, target):
        return sum((results-targets)**2) / len(results)

    def derivative(self, results, target):
        return 2*(results-targets) / len(results)

class L2(Loss):
    """
    L^2 metric loss function
    """
    def __repr__(self):
        return "L^2 metric loss function"

    def function(self, results, target):
        return np.sqrt(sum((results-targets)**2))

    def derivative(self, results, target):
        return results / np.sqrt(sum((results-targets)**2))

class Lp(Loss):
    """
    L^p metric loss function
    """
    def __init__(self, p):
        self.p = p
        Loss.__init__(self)

    def __eq__(self, other):
        return (type(self) == type(other)) and (self.p == other.p)

    def __repr__(self):
        return "L^%s metric loss function" % (self.p,)

    def function(self, results, target):
        return np.power(sum(np.power(results-targets, p)), 1/p)

    def derivative(self, results, target):
        return results / np.power(sum(np.power(results-targets, p)), 1/p)

class L2(Lp):
    def __init__(self):
        Lp.__init__(self, 2)

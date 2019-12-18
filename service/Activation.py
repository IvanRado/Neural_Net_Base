import math

class Activation:

    # Returns the value of the activation function
    # Given for a few common cases
    # Used in both feedforward and back propagation calculations
    def activate(activation_function, x):

        f = 0

        if activation_function == 'sigmoid':
            f = 1 / (1 + math.exp(-x))
        elif activation_function == 'tanh':
            f = (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
        elif activation_function == 'softplus':
            f = math.log(1 + math.exp(x))

        return f

    # Returns the the derivative of the activation function
    # Given for the same activation functions
    # Used in back propagation
    def derivative(activation_function, y, x):
        
        d = 0
        
        if activation_function == 'sigmoid':
            d = y * (1 - y)
        elif activation_function == 'tanh':
            d = 1 - (y*y)
        elif activation_function == 'softplus':
            d = 1 / (1 + math.exp(-x))

        return d


    
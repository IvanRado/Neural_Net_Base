from service.NetworkCalculation import NetworkCalculation
from service.Activation import Activation
class NetworkCost:

    # Calculate loss function (error) for our predictions
    def calculate_loss(instances, nodes, weights, activation_function):

        # Total loss
        J = 0

        # Iterate over the dataset rows
        for i in range(len(instances)):

            # Get the current instance
            instance = instances[i]
            # Assign the weights with random initialization
            nodes = NetworkCalculation.applyForwardPropagation(nodes,weights,instance, activation_function)

            # Get the output of the final node
            predict = nodes[len(nodes) - 1].get_net_value()
            # Get the actual value of the value we are trying to predict
            actual = instance[len(instance) - 1]

            # Calculate the loss or the mean squared error
            loss = (1/2)*(predict - actual)**2

            J = J + loss
            
        # -----------------------

        # Calculate average error for the neural net
        J = J / len(instances)

        return J
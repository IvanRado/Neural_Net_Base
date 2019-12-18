from entity.Weight import Weight
from entity.Node import Node
import math
from service.Activation import Activation

class NetworkCalculation:

    def applyForwardPropagation(nodes, weights, instance, activation_function):

        # Iterate over all nodes
        for j in range(len(nodes)):

            # If the node is a bias unit, give it a value of 1
            if nodes[j].get_is_bias_unit() == True:
                nodes[j].set_net_value(1)

        # instances = [0, 0, 0]


        # Transfer input features to input layer
        # Iterate over the input values
        for j in range(len(instance) - 1):
            
            var = instance[j]
            # Iterate over all nodes
            for k in range(len(nodes)):
                # If in the input layer, assign the nodes value to be the input values
                if j + 1 == nodes[k].get_index():
                    nodes[k].set_net_value(var)
                    break
        # Iterate over all nodes
        for j in range(len(nodes)):
            # If we are dealing with a non-bias unit, calculate the input and output
            if nodes[j].get_level() > 0 and nodes[j].get_is_bias_unit() == False:

                # Initialize net input and net output values
                net_input = 0
                net_output = 0

                # Declare the node we are considering by getting their index
                target_index = nodes[j].get_index()
                # Iterate over the weights list
                for k in range(len(weights)):
                    # If the weights go to the node, add them to the net_input
                    if target_index == weights[k].get_to_index():

                        # Retrieve weight value and its node of origin
                        wi = weights[k].get_value()
                        source_index = weights[k].get_from_index()

                        # Iterate over all nodes; assign weight*value to node being considered's input
                        for m in range(len(nodes)):

                            # If the node matches the node the found weight is coming from
                            if source_index == nodes[m].get_index():
                                # Get the nodes value, multiply it by the weight and add it to the input
                                xi = nodes[m].get_net_value()
                                net_input = net_input + (xi * wi)

                                break

                    # ------------------------------

                
                net_output = Activation.activate(activation_function, net_input)
                nodes[j].set_net_input_value(net_input)
                nodes[j].set_net_value(net_output) 
                            


        return nodes
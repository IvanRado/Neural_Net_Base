from entity.Weight import Weight
import math
import random

class NetworkConnection:
    
    def create_weights(nodes, num_of_features, hidden_nodes):

        # Create the empty weight index
        weights = []

        total_layers = 1 + len(hidden_nodes) + 1 # input layer + hidden layers + output layer

        # Numbering for the weights
        weight_index = 0

        # Iterate over all the layers in the neural net
        for i in range(total_layers - 1):
            # Iterate over all nodes
            for j in range(len(nodes)):
                # Get the level/layer of the current node
                # If the node is in the current layer
                if nodes[j].get_level() == i:

                    # Iterate over all nodes
                    for k in range(len(nodes)):
                        # If the node is in the layer after the layer we are observing
                        if nodes[k].get_level() == (i+1):
                            # If the node is not a bias unit
                            if nodes[k].get_is_bias_unit() == False:
                                # There is a connection from nodes[j] to nodes[k]

                                # ------------------------------------
                                
                                # Randomly initialize the weight values to deviate by up to 2 epsilon from some point
                                range_min = 0
                                range_max = 1
                                init_epsilon = math.sqrt(6)/ (math.sqrt(num_of_features) + 1)

                                rand = range_min + (range_max - range_min) * random.random()
                                rand = rand * (2 * init_epsilon) - init_epsilon


                                # ------------------------------------
                                # Create the weight and assign it all relevant information
                                # Where it goes to and where it originates from
                                # The value it carries and its index
                                weight = Weight()
                                weight.set_weight_index(weight_index)
                                weight.set_from_index(nodes[j].get_index())
                                weight.set_to_index(nodes[k].get_index())
                                weight.set_value(rand)

                                # Increase the weight index and repeat for all connections
                                weight_index += 1
                                weights.append(weight)

                                weight_index += 1

                                print("From " + str(nodes[j].get_label()) + " to "+ str(nodes[k].get_label()) + ": " + str(rand))
                                
        return weights
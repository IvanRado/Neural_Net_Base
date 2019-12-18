from service.NetworkCalculation import NetworkCalculation
from service.Activation import Activation
class NetworkLearning:

    def applyBackPropagation(instances, nodes, weights, learning_rate, activation_function, momentum):

        # Declare number of features as before
        num_of_features = len(instances[0]) - 1

        # Iterate over the dataset
        for i in range(len(instances)):
            # Initialize the neural network with random weights
            nodes = NetworkCalculation.applyForwardPropagation(nodes, weights, instances[i], activation_function)
            # Find the predicted output value for each row
            predicted_value = nodes[len(nodes) - 1].get_net_value()
            # Retrieve the actual value for each row
            actual_value = instances[i][num_of_features]
            # Find small_delta as the error between the predicted and actual values
            small_delta = predicted_value - actual_value
            # Set the error value at the output node as small_delta
            nodes[len(nodes) - 1].set_small_delta(small_delta)

            # Iterate over all nodes, excluding the output node
            for j in range(len(nodes) - 2, num_of_features, -1):
                # Targetted node index
                target_index = nodes[j].get_index()
                # Delta to this node
                sum_small_delta = 0

                # Iterate over all weights
                # Going backwards calculate the error at each step and update weights
                for k in range(len(weights)):

                    # If the current weight is coming from our designated node
                    if weights[k].get_from_index() == target_index:

                        # Get the initial weight value
                        affecting_theta = weights[k].get_value()
                        # Initialize the delta variable
                        affecting_small_delta = 1
                        # The delta we want to find is of the node the weight goes to
                        target_small_delta_index = weights[k].get_to_index()

                        # Iterate over all nodes
                        for m in range(len(nodes)):
                            # If the node we are looking it is connected to the overall considered node through the given weight
                            if nodes[m].get_index() == target_small_delta_index:
                                # Acquire the delta to that node
                                affecting_small_delta = nodes[m].get_small_delta()

                        # ------------------------------------------------
                        # New delta is the weight times the error
                        newly_small_delta = affecting_theta * affecting_small_delta
                        # Sum delta contributions
                        sum_small_delta += newly_small_delta

                # ----------------------
                # Update the nodes error
                nodes[j].set_small_delta(sum_small_delta)

            # ------------------------
            
            previous_derivative = 0
            # Iterate over all weights
            for j in range(len(weights)):

                weight_from_node_value = 0
                weight_to_node_delta = 0
                weight_to_node_value = 0
                weight_to_node_net_input = 0
                # Iterate over all nodes
                for k in range(len(nodes)): 
                    
                    if nodes[k].get_index() == weights[j].get_from_index():

                        weight_from_node_value = nodes[k].get_net_value()

                    if nodes[k].get_index() == weights[j].get_to_index():
                        weight_to_node_value = nodes[k].get_net_value()
                        weight_to_node_net_input = nodes[k].get_net_input_value()
                        weight_to_node_delta = nodes[k].get_small_delta()

                # --------------------------------

                # Calculate the dError/dWeight or dError/dInput
                # Update the weights using gradient descent and momentum
                derivative = weight_to_node_delta * (Activation.derivative(activation_function, weight_to_node_value, weight_to_node_net_input)) * weight_from_node_value
                weights[j].set_value(weights[j].get_value() - learning_rate * derivative + momentum * previous_derivative) 

                previous_derivative = derivative

            # -----------------------             
                                




        return nodes, weights


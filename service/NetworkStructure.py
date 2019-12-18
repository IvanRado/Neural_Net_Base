from entity.Node import Node

class NetworkStructure:

    # Create the nodal structure
    def create_nodes(num_of_features, hidden_nodes):

        # Create the list of nodes we want to return
        # Node index starts at zero and is simply incremented for every created node
        nodes = []
        nodeIndex = 0

        # --------------
        # input layer

        #bias unit
        print("input layer: ", end = '')
        node = Node()
        # The input layer has level 0
        node.set_level(0)
        # Bias units have label "+1"
        node.set_label("+1")
        node.set_index(nodeIndex)
        node.set_is_bias_unit(True)
        nodes.append(node)
        nodeIndex += 1

        print(node.get_label(),"\t", end = '')

        # -----------

        # For every feature we are considering (i.e every variable we think has an impact)
        # we iterate and create a non-bias unit node
        for i in range(num_of_features):
            node = Node()
            node.set_level(0)
            node.set_label("x"+str(i+1))
            node.set_index(nodeIndex)
            node.set_is_bias_unit(False)
            nodes.append(node)
            nodeIndex += 1

            print(node.get_label(),"\t", end = '')

        print("")

        #------------
        # hidden layers

        # Iterate over the hidden_nodes list and create hidden nodes
        # Recall hidden layers still have a bias unit
        for i in range(len(hidden_nodes)):

            print("hidden layer: ", end = '')
            #bias unit
            node = Node()
            # Level is equal to the number of the hidden layer
            # I.e 1st hidden layer has level 1
            node.set_level(i+1)
            node.set_label("+1")
            node.set_index(nodeIndex)
            node.set_is_bias_unit(True)
            nodes.append(node)
            nodeIndex += 1

            print(node.get_label(),"\t", end = '')
            
            # hidden_nodes is a list containing the number of hidden nodes in each layer
            # Iterate over it and create the number of hidden nodes specified
            for j in range(hidden_nodes[i]):
                node = Node()
                node.set_level(i+1)
                node.set_label(("N["+str(i+1)+"]["+str(j+1)+"]"))
                node.set_index(nodeIndex)
                node.set_is_bias_unit(False)
                nodes.append(node)
                nodeIndex += 1

                print(node.get_label(),"\t", end = '')

            print("")

        #-------------------------------------
        #output layer

        # There is only one node in the output layer
        # Recall that the output layer has no bias unit
        node = Node()
        node.set_level(1 + len(hidden_nodes))
        node.set_label("output")
        node.set_index(nodeIndex)
        node.set_is_bias_unit(False)
        nodes.append(node)
        nodeIndex += 1

        print("output layer: ", node.get_label())

        return nodes

class Node:

    # Index tells us the number of the node in sequential order
    def get_index(self):
        return self.__index

    def set_index(self, index):
        self.__index = index

    # Label tells us in what layer the node is found
    def get_label(self):
        return self.__label

    def set_label(self, label):
        self.__label = label

    # Bias units add extra weights but do NOT have connections from the previous layer
    def get_is_bias_unit(self):
        return self.__is_bias_unit

    def set_is_bias_unit(self, is_bias_unit):
        self.__is_bias_unit = is_bias_unit

    # # ------------------------

    # The level of the node tells us in which layer the node resides
    def set_level(self, level):
        self.__level = level

    def get_level(self):
        return self.__level

    # # -----------------------

    # Net input value is the weighted sum
    # of values going into the node
    def set_net_input_value(self, net_input_value):
        self.__net_input_value = net_input_value

    def get_net_input_value(self):
        return self.__net_input_value

    # Net value is the output of the node 
    # (to be multiplied by the) weights for input to the next layer
    def set_net_value(self, net_value):
        self.__net_value = net_value

    def get_net_value(self):
        return self.__net_value

    # ----------------------------------
    # Small delta value is used to update weights
    def set_small_delta(self, small_delta):
        self.__small_delta = small_delta

    def get_small_delta(self):
        return self.__small_delta

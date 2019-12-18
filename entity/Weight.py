class Weight:

    def __init__(self):
        return None

    # The weight index tell us the weight subscripts
    # Also serves to show which connection it is between
    def get_weight_index(self):
        return self.__weight_index

    def set_weight_index(self, weight_index):
        self.__weight_index = weight_index
    
    # The node the weight is originating from (multiplying)
    def get_from_index(self):
        return self.__from_index

    def set_from_index(self, from_index):
        self.__from_index = from_index

    # The node that the weight is feeding into
    def get_to_index(self):
        return self.__to_index

    def set_to_index(self, to_index):
        self.__to_index = to_index

    # The value of the weight 
    def set_value(self,value):
        self.__value = value

    def get_value(self):
        return self.__value
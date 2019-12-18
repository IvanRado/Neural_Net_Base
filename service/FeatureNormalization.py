class FeatureNormalization:

    # Used to ensure that values are within a meaninfgul range
    def normalize(instances, activation_function):

        # find minimum and maximum values for features
        num_of_features = len(instances[0])

        max_items = []
        min_items = []

        # Iterate over the columns (aka the features)
        for j in range(num_of_features): # columns

            # Initialize temporary max and min values
            temp_max = instances[0][j]
            temp_min = instances[0][j]

            # Iterate over the dataset; look through one column at a time
            # (per for loop iteration)
            for i in range(len(instances)):

                # Look at each individual piece of data
                instance = instances[i][j]

                # Update temp_max if instance is bigger
                if instance > temp_max:
                    temp_max = instance
                # Update temp_min if instance is smaller
                if instance < temp_min:
                    temp_min = instance

            # row iteration end
            # Store the max and min for every row
            max_items.append(temp_max)
            min_items.append(temp_min)

            # Reset the temp_min and temp_max
            temp_max = instances[0][j]
            temp_min = instances[0][j]


        
        # normalization logic
        # Iterate over each row of the dataset
        for i in range(len(instances)):
            
            # Iterate over the input values for the dataset
            for j in range(num_of_features):

                # Get the current value
                value = instances[i][j]

                # Get the max value from that row
                maxItem = max_items[j]
                minItem = min_items[j]

                # Check if we are at output
                if j == num_of_features - 1: # output

                    newMax = 1
                    newMin = 0

                # For input features, set limits based on activation function
                else: # input features

                    if activation_function == 'sigmoid':
                        newMax = +4
                        newMin = -4
                    elif activation_function == 'tanh':
                        newMax = 0
                        newMin = 5

                # Normalize the data and reassign
                value = ((newMax - newMin) * ((value - minItem) / (maxItem - minItem))) + newMin
                instances[i][j] = value

        return instances


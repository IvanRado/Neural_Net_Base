if __name__ == "__main__":
	from service.NetworkStructure import NetworkStructure
	from service.NetworkConnection import NetworkConnection
	from service.NetworkCalculation import NetworkCalculation
	from service.NetworkLearning import NetworkLearning
	from service.NetworkCost import NetworkCost
	from service.FeatureNormalization import FeatureNormalization
	from entity.Node import Node
	from entity.Weight import Weight

	# import matplotlib.pyplot as plt
	import copy


	epoch  = 10000 # learning time
	learning_rate = 0.1 # The alpha used to update the weights
	activation_function = 'tanh' # The function that converts the node input into its output
	momentum = 0.1 # Parameter that modifies how quickly we find the minimum

	instances = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]

	# --------------------
	# instances = FeatureNormalization.normalize(instances, activation_function)
	# print("Normalized dataset: ", instances)
	#---------------------

	num_of_features = len(instances[0]) - 1

	hidden_nodes = [3]

	nodes = NetworkStructure.create_nodes(num_of_features, hidden_nodes)

	weights = NetworkConnection.create_weights(nodes, num_of_features, hidden_nodes)

	initial_weights = copy.deepcopy(weights)

	loss_history = []
	for i in range(epoch):
		nodes, weights = NetworkLearning.applyBackPropagation(instances, nodes, weights, learning_rate, activation_function, momentum)

		J = NetworkCost.calculate_loss(instances, nodes, weights, activation_function)
		if i % 100 == 0:
			print(J)
			loss_history.append(str(i) + "\t" + str(J))


	# -------------------- Adaptive learning
	adaptive_loss_history = []
	previous_cost = 0

	for i in range(epoch):
		nodes, initial_weights = NetworkLearning.applyBackPropagation(instances, nodes, initial_weights, learning_rate, activation_function, momentum)

		J = NetworkCost.calculate_loss(instances, nodes, initial_weights, activation_function)
		
		if J < previous_cost:
			learning_rate = learning_rate + 0.1
		else:
			learning_rate = learning_rate - 0.5*learning_rate
		
		previous_cost = J * 1
		
		if i % 100 == 0:
			print(J)
			adaptive_loss_history.append(str(i) + "\t" + str(J))


	# --------------------


	print("Backpropagation is complete")




	for i in range(len(instances)):
		instance = instances[i]
		NetworkCalculation.applyForwardPropagation(nodes, weights, instance, activation_function)

		print("actual: ", instance[len(instance)-1], " - prediction ", nodes[len(nodes) - 1].get_net_value())
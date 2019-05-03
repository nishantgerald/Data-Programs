from decision_tree import DecisionTree
import csv
import numpy as np  # http://www.numpy.org
import ast

# This starter code does not run. You will have to add your changes and
# turn in code that runs properly.

class RandomForest(object):
	num_trees = 0
	decision_trees = []
	# the bootstrapping datasets for trees
	# bootstraps_datasets is a list of lists, where each list in bootstraps_datasets is a bootstrapped dataset.
	bootstraps_datasets = []
	# the true class labels, corresponding to records in the bootstrapping datasets
	# bootstraps_labels is a list of lists, where the 'i'th list contains the labels corresponding to records in
	# the 'i'th bootstrapped dataset.
	bootstraps_labels = []

	def __init__(self, num_trees):
		# Initialization done here
		self.num_trees = num_trees
		self.decision_trees = [DecisionTree() for i in range(num_trees)]

	def _bootstrapping(self, XX, n):
		# Reference: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)
		#
		# TODO: Create a sample dataset of size n by sampling with replacement
		#       from the original dataset XX.
		# Note that you would also need to record the corresponding class labels
		# for the sampled records for training purposes.
			sample=np.random.choice(len(XX),n,replace=False).tolist()
			samples = [] # sampled dataset
			labels = []  # class labels for the sampled records
			for i in sample:
				samples.append(XX[i][:-1])
				labels.append(XX[i][-1])
			return (samples, labels)


	def bootstrapping(self, XX):
        	# Initializing the bootstap datasets for each tree
			for i in range(self.num_trees):
				sample, label = self._bootstrapping(XX, len(XX))
				self.bootstraps_datasets.append(sample)
				self.bootstraps_labels.append(label)

	def fitting(self):
        	# TODO: Train `num_trees` decision trees using the bootstraps datasets
        	# and labels by calling the learn function from your DecisionTree class.
			for i, j in enumerate(self.decision_trees):
				j.learn(self.bootstraps_datasets[i], self.bootstraps_labels[i])
			pass


	def voting(self, X):
			y = []

			for record in X:
				# Following steps have been performed here:
				#   1. Find the set of trees that consider the record as an
				#      out-of-bag sample.
				#   2. Predict the label using each of the above found trees.
				#   3. Use majority vote to find the final label for this recod.
				votes = []
				for i in range(len(self.bootstraps_datasets)):
					dataset = self.bootstraps_datasets[i]
					if record not in dataset:
						OOB_tree = self.decision_trees[i]
						effective_vote = OOB_tree.classify(record)
						votes.append(effective_vote)


				counts = np.bincount(votes)

				if len(counts) == 0:
					# TODO: Special case
					#  Handle the case where the record is not an out-of-bag sample
					#  for any of the trees.
					current_votes=[]
					for i  in self.decision_trees:
						current_votes.append(i.classify(record))
					y.append(np.argmax(np.bincount(current_votes)))
					pass
				else:
					y = np.append(y, np.argmax(counts))
			return y

# DO NOT change the main function apart from the forest_size parameter!
def main():
	X = list()
	y = list()
	XX = list()  # Contains data features and data labels
	numerical_cols = numerical_cols=set([i for i in range(0,43)]) # indices of numeric attributes (columns)
	# Loading data set
	print("reading hw4-data")
	with open("hw4-data.csv") as f:
		for line in csv.reader(f, delimiter=","):
			xline=[]
			for i in range(len(line)):
				if i in numerical_cols:
					xline.append(ast.literal_eval(line[i]))
				else:
					xline.append(line[i])
			X.append(xline[:-1])
			y.append(xline[-1])
			XX.append(xline[:])
	# TODO: Initialize according to your implementation
	y=np.array(y,dtype=int)
	# VERY IMPORTANT: Minimum forest_size should be 10
	forest_size = 10
	# Initializing a random forest.
	randomForest = RandomForest(forest_size)
	# Creating the bootstrapping datasets
	print("creating the bootstrap datasets")
	randomForest.bootstrapping(XX)

	# Building trees in the forest
	print("fitting the forest")
	randomForest.fitting()

	# Calculating an unbiased error estimation of the random forest
	# based on out-of-bag (OOB) error estimate.
	y_predicted = randomForest.voting(X)

	# Comparing predicted and true labels
	results = [prediction == truth for prediction, truth in zip(y_predicted, y)]

	# Accuracy
	accuracy = float(results.count(True)) / float(len(results))

	print("accuracy: %.4f" % accuracy)
	print("OOB estimate: %.4f" % (1-accuracy))


if __name__ == "__main__":
    main()

from util import entropy, information_gain, partition_classes
import numpy as np
import ast

class DecisionTree(object):
	def __init__(self):
    	# Initializing the tree as an empty dictionary or list, as preferred
    	#self.tree = []
		self.tree = {}
		pass

	def learn(self, X, y):
			# TODO: Train the decision tree (self.tree) using the the sample X and labels y
			# You will have to make use of the functions in utils.py to train the tree
			# One possible way of implementing the tree:
			#    Each node in self.tree could be in the form of a dictionary:
			#       https://docs.python.org/2/library/stdtypes.html#mapping-types-dict
			#    For example, a non-leaf node with two children can have a 'left' key and  a
			#    'right' key. You can add more keys which might help in classification
			#    (eg. split attribute and split value)

			if set(y)==set([1]):
				self.tree['class']=1
				return

			elif set(y)==set([0]):
				self.tree['class']=0
				return

			IG_max=-1

			for attr in range(len(X[0])):
				col=np.asarray(X, dtype=np.float32)
				val=col.mean(axis=0).tolist()[attr]

				X_1,X_2,y_1,y_2= partition_classes(X,y,attr,val)
				children_y=[y_1,y_2]

				IG=information_gain(y,children_y)

				if IG>IG_max:
					IG_max=IG
					split_val=val
					split_attribute=attr
					X_left=X_1
					X_right=X_2
					y_left=y_1
					y_right=y_2

			self.tree['left']=DecisionTree()
			self.tree['right']=DecisionTree()
			self.tree['split_val']=val
			self.tree['split_attribute']=attr
			self.tree['left'].learn(X_left, y_left)
			self.tree['right'].learn(X_right, y_right)
			pass

	def classify(self, record):
			# TODO: classify the record using self.tree and return the predicted label
			root = self.tree
			while 'left' in root:
				split_attribute = root['split_attribute']
				split_val = root['split_val']

				if record[split_attribute] <= split_val:
					root = root['left'].tree
				else:
					root = root['right'].tree
			return root['class']
			pass

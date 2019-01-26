############################
# Problem 1: Python Configuration and Data Loading
############################

import sys
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import svm
from scipy import io

# if sys.version_info[0] < 3:
# 	raise Exception("Python 3 not detected.")
# for data_name in ["mnist", "spam", "cifar10"]:
# 	data = io.loadmat("data/%s_data.mat" % data_name)
# 	print("\nloaded %s data!" % data_name)
# 	fields = "test_data", "training_data", "training_labels"
# 	for field in fields:
# 		print(field, data[field].shape)


############################
# Problem 2: Data Partitioning
############################

# a. For the MNIST dataset, write code that sets aside 10,000
#    training images as a validation set.
data = io.loadmat("data/mnist_data.mat")
training_data = data["training_data"]
training_labels = data["training_labels"]
x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(
    training_data, training_labels, test_size=10000, shuffle=True)
print("\nsplited mnist data!")
print("x_train:", x_train.shape, " x_val:", x_val.shape)
print("y_train:", y_train.shape, " y_val:", y_val.shape)

# b. For the spam dataset, write code that sets aside 20% of the
#    training data as a validation set.
data = io.loadmat("data/spam_data.mat")
training_data = data["training_data"]
training_labels = data["training_labels"]
x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(
    training_data, training_labels, test_size=0.2, shuffle=True)
print("\nsplited spam data!")
print("x_train:", x_train.shape, " x_val:", x_val.shape)
print("y_train:", y_train.shape, " y_val:", y_val.shape)

# c. For the CIFAR-10 dataset, write code that sets aside 5,000
#    training images as a validation set.
data = io.loadmat("data/cifar10_data.mat")
training_data = data["training_data"]
training_labels = data["training_labels"]
x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(
    training_data, training_labels, test_size=5000, shuffle=True)
print("\nsplited cifar10 data!")
print("x_train:", x_train.shape, " x_val:", x_val.shape)
print("y_train:", y_train.shape, " y_val:", y_val.shape)


############################
# Problem 3: Support Vector Machines
############################


############################
# Problem 4: Hyper-parameter Tuning
############################


############################
# Problem 5: K-Fold Cross-Validation
############################


############################
# Problem 6: Kaggle
############################
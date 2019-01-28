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


# # a. For the MNIST dataset, write code that sets aside 10,000
# #    training images as a validation set.
# data = io.loadmat("data/mnist_data.mat")
# training_data = data["training_data"]
# training_labels = data["training_labels"]
# ax_train, ax_val, ay_train, ay_val = sklearn.model_selection.train_test_split(
#     training_data, training_labels, test_size=10000, shuffle=True)
# print("\nsplited mnist data!")
# print("ax_train:", ax_train.shape, " ax_val:", ax_val.shape)
# print("ay_train:", ay_train.shape, " ay_val:", ay_val.shape)


# b. For the spam dataset, write code that sets aside 20% of the
#    training data as a validation set.
data = io.loadmat("data/spam_data.mat")
training_data = data["training_data"]
training_labels = data["training_labels"]
bx_train, bx_val, by_train, by_val = sklearn.model_selection.train_test_split(
    training_data, training_labels, test_size=0.2, shuffle=True)
print("\nsplited spam data!")
print("bx_train:", bx_train.shape, " bx_val:", bx_val.shape)
print("by_train:", by_train.shape, " by_val:", by_val.shape)


# # c. For the CIFAR-10 dataset, write code that sets aside 5,000
# #    training images as a validation set.
# data = io.loadmat("data/cifar10_data.mat")
# training_data = data["training_data"]
# training_labels = data["training_labels"]
# cx_train, cx_val, cy_train, cy_val = sklearn.model_selection.train_test_split(
#     training_data, training_labels, test_size=5000, shuffle=True)
# print("\nsplited cifar10 data!")
# print("cx_train:", cx_train.shape, " cx_val:", cx_val.shape)
# print("cy_train:", cy_train.shape, " cy_val:", cy_val.shape)



############################
# Problem 3: Support Vector Machines
############################


# (a) For the MNIST dataset, use raw pixels as features. Train your
#     model with the following numbers of training examples:
#     100, 200, 500, 1000, 2000, 5000, 10000.
# nums = [100, 200, 500, 1000, 2000, 5000, 10000]
# model = svm.LinearSVC()
# at_score, av_score = [], []
# for i in nums:
#     model.fit(ax_train[:i], ay_train[:i].ravel())
#     t_pred = model.predict(ax_train[:i])
#     v_pred = model.predict(ax_val)
#     at_score.append(sklearn.metrics.accuracy_score(ay_train[:i], t_pred))
#     av_score.append(sklearn.metrics.accuracy_score(ay_val, v_pred))
#
# plt.plot(nums, at_score, 'ro', label="training")
# plt.plot(nums, av_score, 'yo', label="validation")
# plt.xlabel('numbers of training examples')
# plt.ylabel('accuracy_score')
# plt.title('MNIST dataset SVM')
# plt.legend()
# plt.savefig('figure_2a.png')
# plt.close()


# # (b) For the spam dataset, use the provided word frequencies as
# #     features. Train your model with the following numbers of training
# #     examples: 100, 200, 500, 1,000, 2,000, ALL.
# nums = [100, 200, 500, 1000, 2000, len(by_train)]
# model = svm.LinearSVC(max_iter=5000)
# bt_score, bv_score = [], []
# for i in nums:
#     model.fit(bx_train[:i], by_train[:i].ravel())
#     t_pred = model.predict(bx_train[:i])
#     v_pred = model.predict(bx_val)
#     bt_score.append(sklearn.metrics.accuracy_score(by_train[:i], t_pred))
#     bv_score.append(sklearn.metrics.accuracy_score(by_val, v_pred))
#
# plt.plot(nums, bt_score, 'ro', label="training")
# plt.plot(nums, bv_score, 'yo', label="validation")
# plt.xlabel('numbers of training examples')
# plt.ylabel('accuracy_score')
# plt.title('spam dataset SVM')
# plt.legend()
# plt.savefig('figure_2b.png')
# plt.close()


# # (c) For the CIFAR-10 dataset, use raw pixels as features. Train your model
# #     with the following numbers of training examples: 100, 200, 500, 1000,
# #     2000, 5000.
# nums = [100, 200, 500, 1000, 2000, 5000]
# model = svm.LinearSVC()
# ct_score, cv_score = [], []
# for i in nums:
#     model.fit(cx_train[:i], cy_train[:i].ravel())
#     t_pred = model.predict(cx_train[:i])
#     v_pred = model.predict(cx_val)
#     ct_score.append(sklearn.metrics.accuracy_score(cy_train[:i], t_pred))
#     cv_score.append(sklearn.metrics.accuracy_score(cy_val, v_pred))
#
# plt.plot(nums, ct_score, 'ro', label="training")
# plt.plot(nums, cv_score, 'yo', label="validation")
# plt.xlabel('numbers of training examples')
# plt.ylabel('accuracy_score')
# plt.title('CIFAR-10 dataset SVM')
# plt.legend()
# plt.savefig('figure_2c.png')
# plt.close()



############################
# Problem 4: Hyper-parameter Tuning
############################


# # (a) For the MNIST dataset, find the best C value.
# C = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
# av_score = []
# for param in C:
#     model = svm.LinearSVC(C=param)
#     model.fit(ax_train[:10000], ay_train[:10000].ravel())
#     v_pred = model.predict(ax_val)
#     av_score.append(sklearn.metrics.accuracy_score(ay_val, v_pred))
#
# print("accuracies:", av_score)
# plt.plot(C, av_score, 'yo')
# plt.xscale('log')
# plt.xlabel('C values')
# plt.ylabel('accuracy_score')
# plt.title('MNIST dataset SVM')
# # plt.show()
# plt.savefig('figure_4.png')
# plt.close()



############################
# Problem 5: K-Fold Cross-Validation
############################


# (a) For the spam dataset, use 5-fold cross-validation to find
#     and report the best C value.

# 1. Partition data into K folds
K = 5
val_xset, val_yset = [bx_val], [by_val]
for i in range(K - 2):
    bx_train, bx_val, by_train, by_val = sklearn.model_selection.\
        train_test_split(bx_train, by_train, test_size=len(val_xset[0]), shuffle=True)
    val_xset.append(bx_val)
    val_yset.append(by_val)
val_xset.append(bx_train)
val_yset.append(by_train)

# 2. Train with different C values
C = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
bv_score = []
for param in C:
    total = 0
    for i in range(K):
        model = svm.LinearSVC(C=param)
        model.fit()
        v_pred = model.predict()
        total += sklearn.metrics.accuracy_score(val_yset[i], v_pred)
    bv_score.append(total/K)

# 3. Plot the graph
print("accuracies:", bv_score)
plt.plot(C, bv_score, 'yo')
plt.xscale('log')
plt.xlabel('C values')
plt.ylabel('accuracy_score')
plt.title('spam dataset K-Fold Cross-Validation')
plt.show()
# plt.savefig('figure_4.png')
# plt.close()



############################
# Problem 6: Kaggle
############################
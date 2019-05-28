"""
This is the starter code and some suggested architecture we provide you with. 
But feel free to do any modifications as you wish or just completely ignore 
all of them and have your own implementations.
"""

"""
Reference: https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
"""

from collections import Counter

import numpy as np
from numpy import genfromtxt
import scipy.io
from matplotlib import pyplot as plt
import save_csv


class DecisionTree:

    def __init__(self, max_depth, max_feature=0, random_forest=False):
        """
        TODO: initialization of a decision tree
        """
        self.random_forest = random_forest
        self.max_depth = max_depth
        self.max_feature = max_feature
        self.split_feature, self.split_thresh = None, None
        self.left, self.right = None, None
        self.X, self.y, self.prediction = None, None, None


    @staticmethod
    def entropy(y):
        """
        TODO: implement a method that calculates the entropy given all the labels
        """
        if len(y) == 0:
            return 0
        p = len(np.where(y < 0.5)[0]) / len(y)
        return -p * np.log(p + 1e-10) - (1 - p) * np.log(1 - p + 1e-10)

    @staticmethod
    def information_gain(X, y, thresh):
        """
        TODO: implement a method that calculates information gain given a vector of features
        and a split threshold
        """
        y1, y2 = y[np.where(X <= thresh)[0]], y[np.where(X > thresh)[0]]
        p1, p2 = len(y1) / len(y), len(y2) / len(y)
        entropy = p1 * DecisionTree.entropy(y1) + p2 * DecisionTree.entropy(y2)
        return DecisionTree.entropy(y) - entropy

    @staticmethod
    def gini_impurity(y):
        """
        TODO: implement a method that calculates the gini impurity given all the labels
        """
        p = len(np.where(y < 0.5)[0]) / len(y)
        return 1 - p * p - (1 - p) * (1 - p)

    @staticmethod
    def gini_purification(X, y, thresh):
        """
        TODO: implement a method that calculates reduction in impurity gain given a vector of features
        and a split threshold
        """
        y1, y2 = y[np.where(X <= thresh)[0]], y[np.where(X > thresh)[0]]
        p1, p2 = len(y1) / len(y), len(y2) / len(y)
        entropy = p1 * DecisionTree.gini_impurity(y1) + p2 * DecisionTree.gini_impurity(y2)
        return DecisionTree.gini_impurity(y) - entropy

    def split(self, X, y, idx, thresh):
        """
        TODO: implement a method that return a split of the dataset given an index of the feature and
        a threshold for it
        """
        set1, set2 = np.where(X[:, idx] <= thresh)[0], np.where(X[:, idx] > thresh)[0]
        X1, y1, X2, y2 = X[set1, :], y[set1], X[set2, :], y[set2]
        return X1, y1, X2, y2
    
    def segmenter(self, X, y):
        """
        TODO: compute entropy gain for all single-dimension splits,
        return the feature and the threshold for the split that
        has maximum gain
        """
        max_gain, split_feature, split_thresh = -float("inf"), -1, -1
        for i in range(len(X[0])):
            thresh = np.linspace(np.min(X[:, i]), np.max(X[:, i]), 10)
            for t in thresh:
                gain = self.information_gain(X[:, i], y, t)
                if gain > max_gain:
                    max_gain, split_feature, split_thresh = gain, i, t
        return split_feature, split_thresh

    def segmenter_random(self, X, y):
        chosen_features = np.random.permutation(np.arange(len(X[0])))[:self.max_feature]
        split_feature, split_thresh = self.segmenter(X[:, chosen_features], y)
        return chosen_features[split_feature], split_thresh
    
    def fit(self, X, y):
        """
        TODO: fit the model to a training set. Think about what would be 
        your stopping criteria
        """
        if self.max_depth == 0:
            self.X, self.y = X, y
            self.prediction = np.argmax(np.bincount(y.flat))
        else:
            if self.random_forest:
                self.split_feature, self.split_thresh = self.segmenter_random(X, y)
            else:
                self.split_feature, self.split_thresh = self.segmenter(X, y)
            X1, y1, X2, y2 = self.split(X, y, self.split_feature, self.split_thresh)

            if len(X1) <= 0 or len(X2) <= 0:
                self.max_depth = 0
                self.X, self.y = X, y
                self.prediction = np.argmax(np.bincount(y.flat))
            else:
                self.left = DecisionTree(self.max_depth - 1)
                self.left.fit(X1, y1)
                self.right = DecisionTree(self.max_depth - 1)
                self.right.fit(X2, y2)

        return self

    def predict(self, X):
        """
        TODO: predict the labels for input data 
        """
        if self.max_depth == 0:
            return self.prediction * np.ones(len(X))
        else:
            set1 = np.where(X[:, self.split_feature] <= self.split_thresh)[0]
            set2 = np.where(X[:, self.split_feature] > self.split_thresh)[0]
            X1, X2 = X[set1, :], X[set2, :]
            y = np.zeros(len(X))
            y[set1] = self.left.predict(X1)
            y[set2] = self.right.predict(X2)
            return y

    def __repr__(self, features=None, depth=0, indent=4):
        """
        TODO: one way to visualize the decision tree is to write out a __repr__ method
        that returns the string representation of a tree. Think about how to visualize 
        a tree structure. You might have seen this before in CS61A.
        """
        if self.prediction is None:
            self.right.__repr__(features=features, depth=depth + 1)
            print(" " * depth * indent, features[self.split_feature], "<=", self.split_thresh)
            self.left.__repr__(features=features, depth=depth + 1)
        else:
            print(" " * depth * indent, 'leaf:', self.prediction)


class RandomForest():
    
    def __init__(self, max_depth, max_feature, num):
        """
        TODO: initialization of a random forest
        """
        self.num = num
        self.forest = [DecisionTree(max_depth, max_feature=max_feature, random_forest=True) for n in range(num)]

    def fit(self, X, y):
        """
        TODO: fit the model to a training set.
        """
        for n in range(self.num):
            indices = np.random.permutation(np.arange(len(X[0])))
            self.forest[n].fit(X[indices, :], y[indices])
        return self
    
    def predict(self, X):
        """
        TODO: predict the labels for input data 
        """
        y = [self.forest[n].predict(X) for n in range(self.num)]
        return np.array(np.round(np.mean(y, axis=0)), dtype=np.bool)

def clean_titanic(data):

    # fill missing value with the most common value of the feature
    for i in range(data.shape[1]):
        common = Counter(data[:, i]).most_common()
        for w, f in common:
            if w != b'':
                substitute = w
                break
        data[data[:, i] == b'', i] = substitute

    # column 1: sex
    data[data == b'female'] = 0
    data[data == b'male'] = 1

    # column 5: ticket
    data[:, 5] = [hash(i) for i in data[:, 5]]
    # column 7:
    data[:, 7] = [hash(i) for i in data[:, 7]]

    # column 8:
    data[data[:, 8] == b'S', 8] = 0
    data[data[:, 8] == b'Q', 8] = 1
    data[data[:, 8] == b'C', 8] = 2

    # speed up the process
    # data = np.delete(data, 5, axis=1)
    # data = np.delete(data, 6, axis=1)
    return np.array(data).astype(np.float)


if __name__ == "__main__":
    # dataset = "titanic"
    dataset = "spam"

    if dataset == "titanic":
        # Load titanic data       
        path_train = 'datasets/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None)
        path_test = 'datasets/titanic/titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None)
        y = data[1:, 0]  # label = survived
        class_names = ["Died", "Survived"]
        
        # TODO: preprocess titanic dataset
        # Notes: 
        # 1. Some data points are missing their labels
        # 2. Some features are not numerical but categorical
        # 3. Some values are missing for some features

        # deleted missing labels
        labeled = np.where(y != b'')[0]
        y = np.array(y[labeled], dtype=np.int)
        X = clean_titanic(data[1:, 1:][labeled, :])
        Z = clean_titanic(test_data[1:, :])
        features = [
            "pclass", "sex", "age", "sibsp", "parch", "ticket",
            "fare", "cabin", "embarked"
        ]

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription",
            "creative", "height", "featured", "differ", "width", "other",
            "energy", "business", "message", "volumes", "revision", "path",
            "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
            "square_bracket", "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = 'datasets/spam-dataset/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]
         
    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)
    
    """
    TODO: train decision tree/random forest on different datasets and perform the tasks 
    in the problem
    """

    ###########
    # 2.5.3
    ###########

    # Split training and validation data
    data_size = len(X)
    slice_idx = int(data_size * 0.2)
    indices = np.random.permutation(data_size)
    x_val, y_val = X[indices][:slice_idx], y[indices][:slice_idx]
    x_train, y_train = X[indices][slice_idx:], y[indices][slice_idx:]

    # depth, val_accracy = [], []
    # for i in range(1, 41):
    #     print(i)
    #     tree = DecisionTree(i)
    #     tree.fit(x_train, y_train)
    #     pred_train = tree.predict(x_train)
    #     pred_val = tree.predict(x_val)
    #     depth.append(i)
    #     val_accracy.append(np.sum(pred_val == y_val) / len(y_val))

    # depth, val_accracy = [], []
    # for i in range(1, 41):
    #     print(i)
    #     forest = RandomForest(i, int(len(X[0]) ** .5), 10)
    #     forest.fit(x_train, y_train)
    #     pred_train = forest.predict(x_train)
    #     pred_val = forest.predict(x_val)
    #     depth.append(i)
    #     val_accracy.append(np.sum(pred_val == y_val) / len(y_val))

    # # plot the figure
    # plt.plot(depth, val_accracy)
    # plt.xlabel("tree depth")
    # plt.ylabel("validation accuracy")
    # plt.show()

    tree = DecisionTree(4)
    tree.fit(x_train, y_train)
    pred_train = tree.predict(x_train)
    pred_val = tree.predict(x_val)
    print('Training Accuracy', np.sum(pred_train == y_train) / len(y_train))
    print('Validation Accuracy', np.sum(pred_val == y_val) / len(y_val))
    tree.__repr__(features=features)

    # forest = RandomForest(5, int(len(X[0])**.5), 10)
    # forest.fit(x_train, y_train)
    # pred_train = forest.predict(x_train)
    # pred_val = forest.predict(x_val)
    # print('Random Forest Training accuracy', np.sum(pred_train == y_train) / len(y_train))
    # print('Random Forest Validation accuracy', np.sum(pred_val == y_val) / len(y_val))

    # save prediction results
    predictions = tree.predict(Z)
    save_csv.results_to_csv(predictions)

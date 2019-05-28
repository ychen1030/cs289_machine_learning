
###############################################################
# Q4: Wine Classification with Logistic Regression
###############################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy import io, cluster
import save_csv
from sklearn import preprocessing

def sigmoid(x):
    """
    A numerically stable version of the logistic sigmoid function.
    Reference: CS 282A Assignment#2
    """
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)

def data_split(training_data, training_labels):
    data_size = len(training_data)
    indices = np.random.permutation(data_size)
    x_train, y_train = training_data[indices][:4800], training_labels[indices][:4800]
    x_val, y_val = training_data[indices][4800:], training_labels[indices][4800:]
    return x_train, x_val, y_train, y_val

def cost_fn(x_train, y_train, w, lambda_):
    noise = 0.000000001
    y_hat = np.dot(x_train, w)
    cost_1 = sum(y_train * np.log(sigmoid(y_hat) + noise))
    cost_2 = sum((1 - y_train) * np.log(1 - sigmoid(y_hat) + noise))
    total_cost = lambda_ * np.linalg.norm(w, ord=2) - (cost_1 + cost_2)
    return total_cost

# load data; split into training and validation sets
data = io.loadmat("data.mat")
training_data, training_labels = data["X"], data["y"]
training_data = preprocessing.scale(training_data)
x_train, x_val, y_train, y_val = data_split(training_data, training_labels)
print("x_train:", x_train.shape, " x_val:", x_val.shape)
print("y_train:", y_train.shape, " y_val:", y_val.shape)


###############################################################
# 1) batch gradient descent
###############################################################
learning_rate, lambda_, iter_ = 0.000001, 0.001, 1000
w = np.zeros((x_train.shape[1], 1))
lost = []
for i in range(iter_):
    cost = cost_fn(x_train, y_train, w, lambda_)
    lost.append(cost)

    y_hat = np.dot(x_train, w)
    grad = 2 * lambda_ * w - np.dot(x_train.T, (y_train - sigmoid(y_hat)))
    w -= learning_rate * grad

plt.plot(range(iter_), lost)
plt.ylabel("loss")
plt.xlabel("# of iterations")
plt.show()


###############################################################
# 2) stochastic gradient descent
###############################################################
learning_rate, lambda_, iter_ = 0.000001, 0.001, 1000
w = np.zeros((x_train.shape[1], 1))
lost = []
for i in range(iter_):
    cost = cost_fn(x_train, y_train, w, lambda_)
    lost.append(cost)

    idx = np.random.randint(x_train.shape[0])
    y_hat = np.dot(x_train[idx:], w)
    grad = 2 * lambda_ * w - np.dot(x_train[idx:].T, (y_train[idx:] - sigmoid(y_hat)))
    w -= learning_rate * grad

plt.plot(range(iter_), lost)
plt.ylabel("loss")
plt.xlabel("# of iterations")
plt.show()


###############################################################
# 3) decaying learning rate
###############################################################
learning_rate, lambda_, iter_ = 0.000001, 0.001, 1000
w = np.zeros((x_train.shape[1], 1))
lost = []
for i in range(iter_):
    alpha = learning_rate / (i + 1)
    cost = cost_fn(x_train, y_train, w, lambda_)
    lost.append(cost)

    idx = np.random.randint(x_train.shape[0])
    y_hat = np.dot(x_train[idx:], w)
    grad = 2 * lambda_ * w - np.dot(x_train[idx:].T, (y_train[idx:] - sigmoid(y_hat)))
    w -= alpha * grad

plt.plot(range(iter_), lost)
plt.ylabel("loss")
plt.xlabel("# of iterations")
plt.show()


###############################################################
# 4) Kaggle
# Kaggel name: Yingying Chen
# Best score: 0.86577
###############################################################

# a. I tried different lambda, and 0.01 seems to have a best results
# b. I tried increase the number of iterations, and 1500 seems to be sufficiently large
# c. I tried different alpha, and 0.00001 provides the best results
# d. Use stochastic gradient descent to get faster computation
# e. Use decaying learning rate
# f. Scale the data so that it has zero mean and unit variance

def predict(w, x_val, threshold=0.5):
    prob = sigmoid(np.dot(x_val, w))
    results = []
    for p in prob:
        if p < threshold:
            results.append(0)
        else:
            results.append(1)
    return np.array(results)


learning_rate, lambda_, iter_ = 0.00001, 0.01, 1500
w = np.zeros((x_train.shape[1], 1))
lost = []

for i in range(iter_):
    alpha = learning_rate / (i + 1)
    cost = cost_fn(x_train, y_train, w, lambda_)
    lost.append(cost)

    idx = np.random.randint(x_train.shape[0])
    y_hat = np.dot(x_train[idx:], w)
    grad = 2 * lambda_ * w - np.dot(x_train[idx:].T, (y_train[idx:] - sigmoid(y_hat)))
    w -= alpha * grad

pred = predict(w, x_val)
accuracy = sum(pred == y_val)/pred.shape[0]
print("val accuracy", accuracy)

test_data_norm = preprocessing.scale(data["X_test"])
predictions = predict(w, test_data_norm)
save_csv.results_to_csv(predictions)



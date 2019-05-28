
###############################################################
# Q2: Isocontours of Normal Distributions
# plot the isocontours of following functions
# Reference: https://docs.scipy.org/doc/scipy-0.14.0/reference/
#            generated/scipy.stats.multivariate_normal.html
###############################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


# part a
x, y = np.mgrid[-1.5:3.5:.01, -2.5:4:.01]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
rv = multivariate_normal([1, 1], [[1, 0], [0, 2]])
plt.contourf(x, y, rv.pdf(pos))
plt.colorbar()
plt.show()


# part b
x, y = np.mgrid[-4:2:.01, -2.5:5.5:.01]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
rv = multivariate_normal([-1, 2], [[2, 1], [1, 3]])
plt.contourf(x, y, rv.pdf(pos))
plt.colorbar()
plt.show()


# part c
x, y = np.mgrid[-2.5:4.5:.01, -2.5:4:.01]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
rv1 = multivariate_normal([0, 2], [[2, 1], [1, 1]])
rv2 = multivariate_normal([2, 0], [[2, 1], [1, 1]])
plt.contourf(x, y, rv1.pdf(pos) - rv2.pdf(pos))
plt.colorbar()
plt.show()


# part d
x, y = np.mgrid[-2.5:4.5:.01, -2.5:4:.01]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
rv1 = multivariate_normal([0, 2], [[2, 1], [1, 1]])
rv2 = multivariate_normal([2, 0], [[2, 1], [1, 3]])
plt.contourf(x, y, rv1.pdf(pos) - rv2.pdf(pos))
plt.colorbar()
plt.show()


# part e
x, y = np.mgrid[-4:4.5:.01, -4:3:.01]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
rv1 = multivariate_normal([1, 1], [[2, 0], [0, 1]])
rv2 = multivariate_normal([-1, -1], [[2, 1], [1, 2]])
plt.contourf(x, y, rv1.pdf(pos) - rv2.pdf(pos))
plt.colorbar()
plt.show()


###############################################################
# Q3: Eigenvectors of the Gaussian Covariance Matrix
###############################################################
np.random.seed(0)
x1 = np.random.normal(3, 3, 100)
x2 = np.random.normal(4, 2, 100)
sample = np.array([np.array((x, x * 0.5 + y)) for (x, y) in zip(x1, x2)])


# part a
mean = np.mean(sample, axis=0)
print('the mean of the sample is ', mean)


# part b
cov = np.cov(sample.T)
print('the covariance of the sample is ', cov)


# part c
w, v = np.linalg.eig(cov)
print('the eigenvalues and eigenvectors of this covariance matrix is ', w, v)


# part d
plt.scatter(sample[:, 0], sample[:, 1])
plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.xlabel("X1")
plt.ylabel("X2")

ax = plt.axes()
ax.arrow(mean[0], mean[1], v[0][0] * w[0], v[1][0] * w[0], head_width=0.3)
ax.arrow(mean[0], mean[1], v[0][1] * w[1], v[1][1] * w[1], head_width=0.3)
plt.show()


# part e
rotation = np.dot(v.T, (sample - mean).T)
plt.scatter(rotation[0, :], rotation[1, :])
plt.xlim(-15, 15)
plt.ylim(-15, 15)
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()


###############################################################
# Q7: Gaussian Classifiers for Digits and Spam
###############################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy import io, cluster
import LDA
import QDA

# part a
data = io.loadmat("mnist-data/mnist_data.mat")
training_data = data["training_data"]
training_labels = data["training_labels"]
training_data_norm = cluster.vq.whiten(training_data)

fitted = {}
for digit in np.unique(training_labels):
    indices = (training_labels == digit).flatten()
    class_data = training_data_norm[indices]
    mean = np.mean(class_data, axis=0)
    cov = np.cov(class_data, rowvar=False, bias=False)
    fitted[digit] = [mean, cov]


# part b
indices = (training_labels == np.array(training_labels[-1])).flatten()
print training_labels[-1]
class_data = training_data_norm[indices]
corrcoef = np.corrcoef(class_data, rowvar=False)
corrcoef = np.abs(corrcoef)
plt.imshow(corrcoef)
plt.colorbar()
plt.show()


# part c
# Reference: https://github.com/scikit-learn/scikit-learn/blob/7389dba/sklearn/discriminant_analysis.py

# First, split the data into training and validation sets
data_size = len(training_data_norm)
indices = np.random.permutation(data_size)
x_val, y_val = training_data_norm[indices][:10000], training_labels[indices][:10000]
x_train, y_train = training_data_norm[indices][10000:], training_labels[indices][10000:]
y_val.flatten()

nums = [100, 200, 500, 1000, 2000, 5000, 10000, 30000, 50000]
model1, model2 = LDA.LDA(), QDA.QDA()
lda_score, qda_score = [], []
for i in nums:
    model1.fit(x_train[:i], y_train[:i])
    model2.fit(x_train[:i], y_train[:i])
    lda_pred = model1.predict(x_val)
    qda_pred = model2.predict(x_val)
    lda_err = 1 - np.sum(lda_pred == y_val)/y_val.shape[0]
    lda_score.append(lda_err)
    qda_err = 1 - np.sum(qda_pred == y_val)/y_val.shape[0]
    qda_score.append(qda_err)

print(lda_score, qda_score)
plt.plot(nums, lda_score, 'ro', label="LDA")
plt.plot(nums, qda_score, 'yo', label="QDA")
plt.xlabel('numbers of training examples')
plt.ylabel('error rate')
plt.title('mnist dataset')
plt.legend()
plt.show()


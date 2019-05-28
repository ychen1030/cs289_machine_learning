import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import io, cluster, stats
import math

class QDA:

    def fit(self, X, y):
        self.classes = np.unique(y)
        n_samples, n_classes = X.shape
        self.params, self.avg_cov = [], np.zeros((n_classes, n_classes))

        for i in self.classes:
            indices = (y == i).flatten()
            class_data = X[indices]
            mean = np.mean(class_data, axis=0)
            cov = np.dot((class_data - mean).T, (class_data - mean))
            self.avg_cov = np.add(self.avg_cov, cov)
            scaled_cov = cov / len(class_data)
            self.params.append((i, mean, scaled_cov, len(class_data)/n_samples))
        self.avg_cov = self.avg_cov / n_samples
        return self

    def predict(self, X):
        pred = np.array([
            (stats.multivariate_normal.logpdf(
                X, allow_singular=True, cov=cov, mean=mean) + math.log(j))
            for (i, mean, cov, j) in self.params])
        return self.classes[np.argmax(pred, axis=0)].reshape((-1, 1))




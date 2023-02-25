import numpy as np


class LogisticRegression:

    def __init__(self, penalty="l2"):
        self.w = None
        err_msg = "penalty must be 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in ["l2", "l1"], err_msg

    def sigmoid(self, x):
        """The logistic sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y, tol=1e-4, max_iter=1000):
        """
        Fit the regression coefficients via gradient descent or other methods
        """
        self.w = np.ones_like(X[0])
        loss_list = []
        for i in range(int(max_iter)):
            w_d = 0
            loss = 0
            for j in range(y.shape[0]):
                w_d = w_d + (X[j] * (y[j] - self.sigmoid(np.dot(self.w, X[j]))))
                if i % 100 == 0:
                    loss = loss - y[j] * np.dot(self.w, X[j]) + np.log(1 + np.exp(np.dot(self.w, X[j])))
            if i % 100 == 0:
                loss_list.append(loss)
            self.w = self.w + tol * w_d

        return loss_list

    def predict(self, X):
        """
        Use the trained model to generate prediction probabilities on a new
        collection of data points.
        """
        y_predict = np.zeros(X.shape[0])
        y_predict[self.sigmoid((X * self.w).sum(axis=1)).reshape(1, -1).squeeze() > 0.5] = 1
        return y_predict

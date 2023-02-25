from XGBoost import XGBoost
from Logistic import LogisticRegression
import random
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt


class Bagging(object):

    def __init__(self, n_estimator=10, attributeRate=0.5, train=True, model_path=None, sample_list_path=None,
                 logistic_path=None, w_train=True):
        # number of base estimators
        self.n_estimator = n_estimator
        # to randomly select attributes
        self.sample_lists = []
        # models of base estimators
        self.models = []
        # logistic estimator used to predict result with result from base estimators
        self.logistic = None
        # the rate of attributes selected randomly
        self.attributeRate = attributeRate
        # whether train base estimators or use saved model
        self.train = train
        # whether the voting model need to be trained
        self.w_train = w_train
        # paths to model of base estimators, sample list, and voting model
        self.model_path = model_path
        self.sample_list_path = sample_list_path
        self.logistic_path = logistic_path

        # get models
        if self.train is False and os.path.exists(self.model_path) and os.path.exists(self.sample_list_path):
            with open(self.model_path, 'rb') as file:
                self.models = pickle.load(file)
            with open(self.sample_list_path, 'rb') as file:
                self.sample_lists = pickle.load(file)
        if self.w_train is False and os.path.exists(self.logistic_path):
            with open(self.logistic_path, 'rb') as file:
                self.logistic = pickle.load(file)

    def fit(self, X, y, rate):
        """train model"""

        # split train set (some for base estimator, some for voting model)
        l = X.shape[0]
        l1 = int(l * rate)  # rate is the percentage of data used in training base estimators
        X0 = X[l1:]
        y0 = y[l1:]
        X = X[:l1]
        y = y[:l1]

        loss_list = []
        fig, ax = plt.subplots()
        plt.grid(True)  # 添加网格
        if self.train is True:

            for i in range(self.n_estimator):
                # randomly select some attributes (attributeRate of total)
                sample_num = int(self.attributeRate * X.shape[1])
                sample_list = [i for i in range(X.shape[1])]
                sample_list = random.sample(sample_list, sample_num)
                # print(sample_list)
                self.sample_lists.append(sample_list)

                # train models of base estimator
                X1 = X[:, sample_list]
                model = XGBoost(n_trees=20, depth=3, min_impurity=1e-7, tol=1, lam=30, ax=ax, fig=fig)
                loss_list = model.fit(X1, y, loss_list)
                self.models.append(model)

            # save model
            modelsFile = "../model/{}_{}_models.pkl".format(self.n_estimator, int(self.attributeRate * 10))
            sampleListsFile = "../model/{}_{}_sampleLists.pkl".format(self.n_estimator, int(self.attributeRate * 10))

            with open(modelsFile, "wb") as file:
                pickle.dump(self.models, file)
            with open(sampleListsFile, "wb") as file:
                pickle.dump(self.sample_lists, file)

        if self.w_train is True:
            # train voting model
            self.logistic = self.fit_w(X0, y0)
            logisticFile = "../model/{}_{}_logistic.pkl".format(self.n_estimator, int(self.attributeRate * 10))
            with open(logisticFile, "wb") as file:
                pickle.dump(self.logistic, file)

    def fit_w(self, X, y):
        """train voting model"""
        y1 = []
        # make the result of n base estimators into new attributes
        for i in range(self.n_estimator):
            sample_list = self.sample_lists[i]
            X1 = X[:, sample_list]
            model = self.models[i]
            y_pred = model.predict(X1)
            y1.append(y_pred)
        y1 = np.array(y1)
        # print("y1_shape:", y1.shape)
        y1 = y1.transpose()
        # print("y1_shape:", y1.shape)
        # voting model is logistic regression
        model = LogisticRegression()
        model.fit(y1, y)
        return model

    def predict(self, X):
        """predict result"""
        y1 = []
        # print("sample_lists_size:", len(self.sample_lists))
        # print("model_size:", len(self.models))

        # make the result of n base estimators into new attributes
        for i in range(self.n_estimator):
            sample_list = self.sample_lists[i]
            X1 = X[:, sample_list]
            model = self.models[i]
            y_pred = model.predict(X1)
            # print(y_pred)

            y1.append(y_pred)

        y1 = np.array(y1)
        # print("y1_shape:", y1.shape)
        y1 = y1.transpose()
        # print("y1_shape:", y1.shape)
        y = self.logistic.predict(y1)

        return y

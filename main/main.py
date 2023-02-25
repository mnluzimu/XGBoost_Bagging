import numpy as np
import time
import math
from dataset import Canser
from Bagging.bagging import Bagging


def shuffleData(X, y):
    """randomly shuffle data"""
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]


def train_test_split(X, y, test_size=0.5, shuffle=True):
    """split test and train set"""
    if shuffle:
        X, y = shuffleData(X, y)

    split_idx = len(y) - int(len(y) * test_size)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    precisions = []
    runtimes = []
    # for i in range(10):
    # get data
    data = Canser()
    X, y = data.getXy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    # prepare model
    start = time.time()
    model = Bagging(n_estimator=10, attributeRate=0.6, train=True, w_train=True, model_path="../model/40_6_models.pkl", sample_list_path="../model/40_6_sampleLists.pkl", logistic_path="../model/40_6_logistic.pkl")
    model.fit(X_train, y_train, rate=0.6)
    y_pred = model.predict(X_test)

    # predict on test set
    # print("y_test:", y_test)
    # print("y_pred:", y_pred)
    precision = np.sum(y_pred == y_test) / len(y_test)
    print('precision:', np.sum(y_pred == y_test) / len(y_test))
    precisions.append(precision)

    # run time
    runtime = time.time() - start
    print("run time:", time.time() - start)
    runtimes.append(runtime)

    print(precisions)
    print(runtimes)
    print("ave precision:", np.average(np.array(precisions)))
    print("average runtime:", np.average(np.array(runtimes)))




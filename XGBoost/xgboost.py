import numpy as np
import matplotlib.pyplot as plt
import random


def divide_on_feature(X, feature_i, threshold):
    """divide by feature threshold"""
    X_1 = np.array([sample for sample in X if sample[feature_i] <= threshold])
    X_2 = np.array([sample for sample in X if sample[feature_i] > threshold])

    return X_1, X_2


class XGBoostTreeNode:

    def __init__(self, feature_i=None, threshold=None,
                 value=None, right_branch=None, left_branch=None):
        self.feature_i = feature_i
        self.threshold = threshold
        self.value = value
        self.right_branch = right_branch
        self.left_branch = left_branch


class Sigmoid:
    """sigmoid function class"""
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))


class LogisticLoss:
    """logistic loss"""
    def __init__(self):
        sigmoid = Sigmoid()
        self._func = sigmoid
        self._grad = sigmoid.gradient

    def loss(self, y, y_pred):
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        p = self._func(y_pred)
        return - (y * np.log(p) + (1 - y) * np.log(1 - p))

    def gradient(self, y, y_pred):
        p = self._func(y_pred)
        return -(y - p)

    def hess(self, y, y_pred):
        p = self._func(y_pred)
        return p * (1 - p)


class XGBoostRegressionTree(object):
    """XGBoost Regression Tree"""
    def __init__(self, min_samples_split=2, min_impurity=1e-7,
                 tree_depth=float("inf"), loss=None, lam=None):
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.tree_depth = tree_depth
        self.loss = loss
        self.lam = lam

    def split_y(self, y):
        """split y"""
        col = int(np.shape(y)[1] / 2)
        y, y_pred = y[:, :col], y[:, col:]
        return y, y_pred

    def gain(self, y, y_pred):
        """gain of a split"""
        nominator = np.power((self.loss.gradient(y, y_pred)).sum(), 2)
        denominator = self.loss.hess(y, y_pred).sum()
        return 0.5 * (nominator / denominator + self.lam)

    def gain_by_taylor(self, y, y1, y2):
        """total gain"""
        y, y_pred = self.split_y(y)
        y1, y1_pred = self.split_y(y1)
        y2, y2_pred = self.split_y(y2)

        true_gain = self.gain(y1, y1_pred)
        false_gain = self.gain(y2, y2_pred)
        gain = self.gain(y, y_pred)
        return true_gain + false_gain - gain

    def approximate_update(self, y):
        """produce leaf value"""
        y, y_pred = self.split_y(y)
        gradient = np.sum(self.loss.gradient(y, y_pred), axis=0)
        hessian = np.sum(self.loss.hess(y, y_pred), axis=0)
        update_approximation = - gradient / (hessian + self.lam)
        return update_approximation

    def fit(self, X, y):
        """train model"""
        self.root = self._build_tree(X, y)
        self.loss = None

    def _build_tree(self, X, y, current_depth=0):
        """build tree"""
        largest_impurity = 0
        best_criteria = None
        best_sets = None

        Xy = np.concatenate((X, y), axis=1)

        n_samples, n_features = np.shape(X)

        if n_samples >= self.min_samples_split and current_depth <= self.tree_depth:
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)
                unique_values.sort()

                for i in range(len(unique_values) - 1):
                    threshold = (unique_values[i] + unique_values[i + 1]) / 2

                    Xy1, Xy2 = divide_on_feature(Xy, feature_i, threshold)

                    if len(Xy1) > 0 and len(Xy2) > 0:
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]

                        impurity = self.gain_by_taylor(y, y1, y2)

                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_i": feature_i, "threshold": threshold}
                            best_sets = {
                                "leftX": Xy1[:, :n_features],
                                "lefty": Xy1[:, n_features:],
                                "rightX": Xy2[:, :n_features],
                                "righty": Xy2[:, n_features:]
                            }
        # print("impurity:", largest_impurity)
        if largest_impurity > self.min_impurity:
            right_branch = self._build_tree(best_sets["rightX"], best_sets["righty"], current_depth + 1)
            left_branch = self._build_tree(best_sets["leftX"], best_sets["lefty"], current_depth + 1)
            return XGBoostTreeNode(feature_i=best_criteria["feature_i"], threshold=best_criteria[
                "threshold"], right_branch=right_branch, left_branch=left_branch)

        leaf_value = self.approximate_update(y)
        return XGBoostTreeNode(value=leaf_value)

    def predict_value(self, x, tree=None):

        if tree is None:
            tree = self.root

        if tree.value is not None:
            return tree.value

        feature_value = x[tree.feature_i]

        if feature_value <= tree.threshold:
            branch = tree.left_branch
        else:
            branch = tree.right_branch

        return self.predict_value(x, branch)

    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self.predict_value(x))
        return y_pred


class XGBoost(object):
    """XGBoost class"""
    def __init__(self, n_trees=300, tol=0.01, min_split_n=2,
                 min_impurity=1e-7, depth=2, lam=1, ax=None, fig=None):
        self.n_trees = n_trees
        self.tol = tol
        self.min_split_n = min_split_n
        self.min_impurity = min_impurity
        self.depth = depth
        self.loss = LogisticLoss()
        self.trees = []
        self.lam = lam
        self.ax = ax
        self.fig = fig

        for _ in range(n_trees):
            tree = XGBoostRegressionTree(
                min_samples_split=self.min_split_n,
                min_impurity=min_impurity,
                tree_depth=self.depth,
                loss=self.loss,
                lam=self.lam
            )

            self.trees.append(tree)

    def fit(self, X, y, loss_list):
        """train model"""
        m = X.shape[0]
        y = np.reshape(y, (m, -1))
        y_pred = np.zeros(np.shape(y))
        losses = []
        t = []
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0), (1, 1, 0), (0, 1, 1),
                  (1, 0, 1), (0, 0, 0)]
        color = random.sample(colors, 1)


        for i in range(self.n_trees):
            tree = self.trees[i]
            y_and_pred = np.concatenate((y, y_pred), axis=1)
            tree.fit(X, y_and_pred)
            update_pred = tree.predict(X)
            update_pred = np.reshape(update_pred, (m, -1))
            y_pred += update_pred * self.tol
            loss = np.sum(self.loss.loss(y, y_pred))
            losses.append(loss)
            t.append(i + 1)
            self.ax.cla()  # clear plot
            for loss_pre, color_pre in loss_list:
                t_pre = np.arange(20) + 1
                line_pre = self.ax.plot(t_pre, loss_pre, color=color_pre, lw=1)[0]  # draw line chart
                line_pre.set_xdata(t_pre)
                line_pre.set_ydata(loss_pre)

            line = self.ax.plot(t, losses, color=color[0], lw=1)[0]  # draw line chart
            line.set_xdata(t)
            line.set_ydata(losses)
            plt.title("loss curve")
            plt.xlabel("rounds")
            plt.ylabel("loss")
            self.ax.set_ylim([150, 210])
            self.ax.set_xlim([0, 21])

            plt.pause(0.1)

            # print(loss, end=" ")
        # print()
        loss_list.append((losses, color[0]))
        print(losses)
        return loss_list
        # plt.show()

    def predict(self, X):
        """predict test set"""
        y_pred = None
        m = X.shape[0]
        for tree in self.trees:
            update_pred = tree.predict(X)
            update_pred = np.reshape(update_pred, (m, -1))
            if y_pred is None:
                y_pred = np.zeros_like(update_pred)
            y_pred += self.tol * update_pred

        y_pred = y_pred.flatten()
        y_pred[y_pred < 0.5] = 0
        y_pred[y_pred > 0.5] = 1

        return y_pred

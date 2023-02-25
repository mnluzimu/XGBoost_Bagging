import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Canser(object):
    def __init__(self, path='../data/cancer/wpbc.data'):
        """read the dataset and delete useless attributes"""
        self.path = path
        df = pd.read_csv(path, sep=',', names=['name{}'.format(i) for i in range(41)])
        df.drop('name0', axis=1, inplace=True)  # drop the ID
        df.drop('name34', axis=1, inplace=True)
        df.drop('name33', axis=1, inplace=True)
        df.drop('name35', axis=1, inplace=True)
        df.drop('name36', axis=1, inplace=True)
        df.drop('name37', axis=1, inplace=True)
        df.drop('name38', axis=1, inplace=True)
        df.drop('name39', axis=1, inplace=True)
        df.drop('name40', axis=1, inplace=True)
        df.name1=df.name1.map({'N': 1, 'R': 0})
        df.info()
        data = np.array(df).astype(np.float32)
        self.X = data[:, 1:]
        self.y = data[:, 0].astype(np.int)
        # print("X shape:", self.X.shape)
        # print("y shape:", self.y.shape)
        # print("first 20 X's:")
        # print(self.X[:20])
        # print("first 20 y's:")
        # print(self.y[:20])
        self.plot_features()

    def getXy(self):
        """return X y"""
        return self.X, self.y

    def plot_features(self):
        X0 = self.X[self.y == 0]
        y0 = np.where(self.y == 0)[0]
        X1 = self.X[self.y == 1]
        y1 = np.where(self.y == 1)[0]
        # plt.figure(figsize=(8, 4))
        # plt.title("distribution of different attributes")
        for i in range(self.X.shape[1]):
            x0 = X0[:, i]
            x1 = X1[:, i]
            # print("x0 shape:", x0.shape)
            # print("y0 shape:", y0.shape)
            # print("y1 shape:", y1.shape)
            # print("x1 shape:", x1.shape)
            # plt.subplot(4, 8, i + 1)
            # plt.scatter(y0, x0, color="r", label="recurrent")
            # plt.scatter(y1, x1, color="b", label="nonrecurrent")
            # plt.xlabel("instances")
            # plt.ylabel("value")
            # plt.title("attribute {}".format(i + 1))
            # plt.show()

        # plt.savefig("../images/attributes.png")




if __name__ == '__main__':
    Canser()

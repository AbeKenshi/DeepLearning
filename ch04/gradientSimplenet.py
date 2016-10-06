import sys, os

from ch03.softmaxFunction import softmax
from ch04.crossEntropyError import cross_entropy_error
from ch04.numericalGradient import numerical_gradient

sys.path.append(os.pardir)
import numpy as np


class simpleNet:
    def __init__(self):
        self.W = np.random.rand(2, 3)  # ガウス分布で初期化

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


if __name__ == '__main__':
    x = np.random.rand(2)
    t = np.array([0, 0, 1])

    net = simpleNet()

    f = lambda  w: net.loss(x, t)

    dW = numerical_gradient(f, net.W)
    print(dW)



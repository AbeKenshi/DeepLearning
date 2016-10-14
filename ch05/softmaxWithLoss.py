import numpy as np

from ch03.softmaxFunction import softmax
from ch04.crossEntropyError import cross_entropy_error


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # 損失
        self.y = None  # softmaxの出力
        self.t = None  # 教師データ（one-hot vector）

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 教師データがone-hot-vectorの場合
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx

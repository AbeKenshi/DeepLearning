import matplotlib

matplotlib.use('Qt4Agg')  # ウィンドウをshowしたときに前に出すときに必要
import numpy as np
import matplotlib.pylab as plt


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))


if __name__ == '__main__':
    x = np.array([0.3, 2.9, 4.0])
    y = softmax(x)
    print(y)
    plt.plot(x, y)
    # plt.ylim(-0.1, 1.1)  # y軸の範囲を指定
    plt.show()

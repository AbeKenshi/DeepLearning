import matplotlib

matplotlib.use('Qt4Agg')  # ウィンドウをshowしたときに前に出すときに必要
import numpy as np
import matplotlib.pylab as plt


def identity_function(x):
    return x


if __name__ == '__main__':
    x = np.arange(-5.0, 5.0, 0.1)
    y = identity_function(x)
    plt.plot(x, y)
    # plt.ylim(-0.1, 1.1)  # y軸の範囲を指定
    plt.show()
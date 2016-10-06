import matplotlib

matplotlib.use('Qt4Agg')  # ウィンドウをshowしたときに前に出すときに必要
import numpy as np
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    grad = np.zeros(x)
    grad[x>=0] = 1
    return grad


if __name__ == '__main__':
    x = np.arange(-5.0, 5.0, 0.1)
    y = relu(x)
    plt.plot(x, y)
    plt.ylim(-1.0, 5.0)  # y軸の範囲を指定
    plt.show()
    plt.get_current_fig_manager().window.raise_()

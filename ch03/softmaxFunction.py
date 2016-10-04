import matplotlib

matplotlib.use('Qt4Agg')  # ウィンドウをshowしたときに前に出すときに必要
import numpy as np
import matplotlib.pylab as plt


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c) # オーバーフロー対策
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


if __name__ == '__main__':
    x = np.array([0.3, 2.9, 4.0])
    y = softmax(x)
    print(y)
    plt.plot(x, y)
    # plt.ylim(-0.1, 1.1)  # y軸の範囲を指定
    plt.show()

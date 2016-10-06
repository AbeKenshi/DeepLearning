import numpy as np
import matplotlib

matplotlib.use('Qt4Agg')  # ウィンドウをshowしたときに前に出すときに必要
from ch04.function2 import function_2
from ch04.numericalGradient import numerical_gradient_2d

import matplotlib.pylab as plt


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())
        grad = numerical_gradient_2d(f, x)
        x -= lr * grad

    return x, np.array(x_history)


if __name__ == '__main__':
    init_x = np.array([-3.0, 4.0])

    lr = 1e-1
    step_num = 100
    x, x_history = gradient_descent(function_2, init_x, lr=lr, step_num=step_num)
    print(x)

    plt.plot([-5, 5], [0, 0], '--b')
    plt.plot([0, 0], [-5, 5], '--b')
    plt.plot(x_history[:, 0], x_history[:, 1], 'o')

    plt.xlim(-3.5, 3.5)
    plt.ylim(-4.5, 4.5)
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.show()

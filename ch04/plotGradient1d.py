import matplotlib

from ch04.function1 import function_1
from ch04.tangentLine import tangent_line

matplotlib.use('Qt4Agg')  # ウィンドウをshowしたときに前に出すときに必要
import numpy as np
import matplotlib.pylab as plt

if __name__ == '__main__':
    x = np.arange(0.0, 20.0, 0.1)
    y = function_1(x)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    tf = tangent_line(function_1, 5)
    y2 = tf(x)

    plt.plot(x, y)
    plt.plot(x, y2)
    plt.show()

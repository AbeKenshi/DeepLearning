import matplotlib

matplotlib.use('Qt4Agg')  # ウィンドウをshowしたときに前に出すときに必要
import numpy as np
import matplotlib.pyplot as plt

# データの作成
x = np.arange(0, 6, 0.1)  # 0から6まで0.1刻みで生成
y = np.sin(x)

# グラフの描画
plt.plot(x, y)
plt.show()

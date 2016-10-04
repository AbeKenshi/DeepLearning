import matplotlib

matplotlib.use('Qt4Agg')  # ウィンドウをshowしたときに前に出すときに必要
import matplotlib.pylab as plt
from matplotlib.image import imread

img = imread('../dataset/lena.png') # 画像の読み込み（適切なパスを設定する）
plt.imshow(img)

plt.show()
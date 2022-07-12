from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.utils import get_images

x = y = np.linspace(-2 * np.pi, 2 * np.pi, 224)
xx, yy = np.meshgrid(x, y)
im = np.stack(tuple(np.cos(x) + np.sin(y) for x, y in zip(xx, yy)))

fig = plt.figure(figsize=(9, 4))

# 等高線を作成する。
ax1 = fig.add_subplot(121)
ax1.set_title("contour")
ax1.contourf(x, y, im)
print(x)
# 普通のイメージ
ax2 = fig.add_subplot(122)
ax2.set_title("image")
ax2.imshow(im, cmap="plasma")

print(im.shape)
plt.savefig("sin.png")
# plt.contourf(x, y, im)
# plt.show()
# print(yy)
# y1 = x
# y2 = x**2
# y3 = x**3
# y4 = x**4

# plt.figure(figsize=(5, 4))

# # 余白を設定
# plt.subplots_adjust(wspace=0.4, hspace=0.6)

# # 左上
# plt.subplot(2, 2, 1)
# plt.plot(x, y1)

# # 右上
# plt.subplot(2, 2, 2)
# plt.plot(x, y2)

# # 左下
# plt.subplot(2, 2, 3)
# plt.plot(x, y3)

# # 右下
# plt.subplot(2, 2, 4)
# plt.plot(x, y4)

# plt.show()

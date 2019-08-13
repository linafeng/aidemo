# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt

# 读取图片的数据，存放到img
img = plt.imread('F://pic1.png')
print(img.shape)  # 打印图片的大小

plt.imshow(img)  # 展示图片

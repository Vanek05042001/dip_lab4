# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 22:20:52 2024

@author: Vanya
"""

import sys
sys.path.append('../')
# %matplotlib inline
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from utility import util

plt.rcParams["figure.figsize"] = [6, 4]

image1 = cv.imread('lab4.jpg')
rgb_image1 = cv.cvtColor(image1, cv.COLOR_BGR2RGB)
hsv_image1 = cv.cvtColor(image1, cv.COLOR_BGR2HSV)
gray_image1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)

channels = [0]
histSize = [256]
range = [0, 256]

gs = plt.GridSpec(2, 2)
plt.figure(figsize=(10, 8))
plt.subplot(gs[0])
plt.imshow(image1, cmap='gray')
plt.subplot(gs[2])
plt.hist(image1.reshape(-1), 256, range)
plt.show()

# Бинаризация
threshold = 190
image = gray_image1

ret, thresh1 = cv.threshold(image, threshold, 255, cv.THRESH_TRUNC)

plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.imshow(thresh1, 'gray', vmin=0, vmax=255)
plt.title('Бинаризация')
plt.xticks([])
plt.yticks([])
plt.show()

cv.imwrite('new2.jpg', thresh1)

# Выделение границ
outImageDepth = cv.CV_16S  

gaussian33 = cv.GaussianBlur(thresh1, (3, 3), 0)
gaussian55 = cv.GaussianBlur(thresh1, (5, 5), 0)

laplace = cv.Laplacian(thresh1, outImageDepth, ksize=3)

log = cv.Laplacian(gaussian33, outImageDepth, ksize=3)
log = cv.convertScaleAbs(log)

# Вывод
plt.figure(figsize=(15, 8))
gs = plt.GridSpec(1, 3)


kernel1 = np.asarray([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
kernel2 = np.asarray([[-0.25, -0.25, -0.25], [-0.25, 3, -0.25], [-0.25, -0.25, -0.25]])
kernel3 = np.asarray([[0, -0.25, 0], [-0.25, 2, -0.25], [0, -0.25, 0]])

result_image = cv.filter2D(laplace, -1, kernel2)

plt.subplot(gs[0])
plt.xticks([]), plt.yticks([])
plt.title('Результат')
plt.imshow(result_image, cmap='gray')
plt.show()

# Сохранение в файл
cv.imwrite('new.jpg', result_image)

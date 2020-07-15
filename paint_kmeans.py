# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 01:14:27 2020

@author: VINAY KE
"""

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
img = cv.imread('obama.jpg')
img_copy = img.copy()
img_shape = img.shape[:2]
img = cv.resize(img, (512,512))
img = cv.GaussianBlur(img,(7,7),0)
img = cv.resize(img, img_shape[::-1])

#img = cv.resize(img, (int(shape[0]*0.5), int(shape[1]*0.5)))
Z = img.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 8
ret,label,center=cv.kmeans(Z,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
res2 = cv.GaussianBlur(res2,(5,5),0)

res2 = cv.addWeighted(img_copy,0.4,res2,0.6,0)

plt.subplot(121)
plt.axis("off")
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.subplot(122)
plt.axis("off")
plt.imshow(cv.cvtColor(res2, cv.COLOR_BGR2RGB))
plt.show()
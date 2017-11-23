import numpy as np
from PIL import Image
from scipy.spatial import distance_matrix
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

img1= Image.open('F:/college/9th term/pattern/BSR/BSDS500/data/images/train/2092.jpg')
img1.load()
data=np.asarray(img1)

nrow=data.shape[0]
ncol=data.shape[1]
center_row=np.random.choice(nrow,3,replace=False)
center_col=np.random.choice(ncol,3,replace=False)
centroids=np.zeros((3,3))
centroids_old=np.zeros((3,3))
dataVector=data.reshape(154401,3)

for i in range(center_row.shape[0]):
    row=center_row[i]
    col=center_col[i]
    centroids[i]=data[row,col]

while (centroids_old != centroids).any():
    centroids_old = centroids.copy()
    distances = np.sqrt(((dataVector - centroids[:, np.newaxis])**2).sum(axis=2))
    cluster_assignements=np.argmin(distances, axis=0)
    centroids=np.array([dataVector[cluster_assignements==k].mean(axis=0) for k in range(centroids.shape[0])])
print(centroids)
print((np.argmin(distances, axis=0)))

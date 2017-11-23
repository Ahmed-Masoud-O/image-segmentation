import numpy as np
import scipy.spatial.distance as dist
from scipy.sparse import csgraph
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import linalg as LA
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

data = np.matrix('2,4;3,3;3,4;5,4;5,6;5,8;6,4;6,5;6,7;7,3;7,4;8,2;9,4;10,6;10,7;10,8;11,5;11,8;12,7;13,6;13,7;14,6;15,4;15,5')
A = kneighbors_graph(data, 3)
simmatrix=A.toarray()
B=csgraph.laplacian(simmatrix, normed=False)
w, v = LA.eigh(B)
v.sort(axis=1)
vK3=v[:,:3]
vk3norm=normalize(vK3)
kmeans=KMeans(n_clusters=3)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
kmeans.fit(vk3norm)
clusters = kmeans.labels_
ax.scatter(vk3norm[:,0],vk3norm[:,1],vk3norm[:,2],c=clusters, edgecolor='k')
plt.show()

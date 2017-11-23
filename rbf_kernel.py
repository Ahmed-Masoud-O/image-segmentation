from sklearn.metrics.pairwise import rbf_kernel as rbf
from scipy.sparse import csgraph
from sklearn.preprocessing import normalize
from helper import *
import matplotlib.pyplot as plt

number_of_clusters = 3
img_name = '2092'
image_data, img = get_image_data(img_name)
#img.show()
print(image_data.shape)
reshaped_data = image_data.reshape(image_data.shape[0]*image_data.shape[1], image_data.shape[2])
print(reshaped_data.shape)
RBF_01 = rbf(reshaped_data, gamma=1)
# RBF_10 = rbf(reshaped_data, gamma=10)
lp_matrix_01 = csgraph.laplacian(RBF_01, normed=False)
# lp_matrix_10 = laplacian_matrix(from_numpy_array(RBF_10)).toarray()
eigens = np.linalg.eigh(lp_matrix_01)
eig_values = eigens[0]
eig_vectors = eigens[1]
np.save('rbf_values3', eig_values)
np.save('rbf_vectors3', eig_vectors)
# eig_values = np.load('rbf_values3.npy')
# eig_vectors = np.load('rbf_vectors3.npy')
normalized_vector = normalize(eig_vectors[:, 0:number_of_clusters])


ground_truth, img = load_image(img_name)
for i in range(ground_truth.shape[1]):
    segment = ground_truth[0, i]
    segments_array = segment[0, 0][0]
segments_array = np.ravel(segments_array)
print(normalized_vector.shape)
assignments = k_means(normalized_vector, number_of_clusters, 1, 1000, number_of_clusters)
data = get_image_data(img_name)
data = data[0]
img.show()
clustered_image = np.zeros([data.shape[0], data.shape[1], 3], dtype=np.uint8)
red_component = np.random.randint(256, size=number_of_clusters)
blue_component = np.random.randint(256, size=number_of_clusters)
green_component = np.random.randint(256, size=number_of_clusters)
k = 0
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        clustered_image[i][j] = [red_component[assignments[k]], green_component[assignments[k]],
                                 blue_component[assignments[k]]]
        k += 1
plt.imshow(clustered_image)
plt.show()

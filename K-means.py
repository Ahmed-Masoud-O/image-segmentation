import numpy as np
from helper import k_means
import matplotlib.pyplot as plt
from helper import load_image
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import v_measure_score
from helper import get_image_data

image_name = '15004'
cluster_numbers = 11
ground_truth, img = load_image(image_name)
for i in range(ground_truth.shape[1]):
    segment = ground_truth[0, i]
    segments_array = segment[0, 0][0]
segments_array = np.ravel(segments_array)
image_data, img = get_image_data(image_name)
assignments = k_means(image_data, cluster_numbers)
img.show()
clustered_image = np.zeros([image_data.shape[0], image_data.shape[1], 3], dtype=np.uint8)
red_component = np.random.randint(256, size=cluster_numbers)
blue_component = np.random.randint(256, size=cluster_numbers)
green_component = np.random.randint(256, size=cluster_numbers)
k = 0
for i in range(image_data.shape[0]):
    for j in range(image_data.shape[1]):
        clustered_image[i][j] = [red_component[assignments[k]], green_component[assignments[k]],
                                 blue_component[assignments[k]]]
        k += 1
plt.imshow(clustered_image)
plt.show()
print("f1 measure")
print(f1_score(segments_array, assignments, average='micro'))
print("conditional entropy")
print(v_measure_score(segments_array, assignments))



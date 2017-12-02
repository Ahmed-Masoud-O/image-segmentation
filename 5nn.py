from helper import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csgraph
from numpy import linalg as LA
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize

cluster_numbers_list = [3, 5, 7, 9, 11]
image_names_list = load()
# image_names_list = ['198054', '299091']
x = 50
print(len(image_names_list))
for z in range(len(image_names_list)):

    image_name = image_names_list[z]
    print("------------------------")
    print("image number = ", x)
    print("------------------------")
    x += 1
    # if x == 83  or x < 83 :
    #     continue
    image_data, img = get_image_data(image_name)
    try:
        eig_vectors = np.load('knn_outputs/knn_vectors_' + image_name + '.npy')
    except:

        reshaped_data = image_data.reshape(image_data.shape[0]*image_data.shape[1], image_data.shape[2])
        sim_graph = kneighbors_graph(reshaped_data, 5)
        sim_matrix = sim_graph.toarray()
        # plt.matshow(sim_matrix)
        # plt.show()
        lap_matrix = csgraph.laplacian(sim_matrix, normed=False)
        eig_values, eig_vectors = LA.eigh(lap_matrix)
        np.save('knn_outputs/knn_vectors_' + image_name, eig_vectors)
    # eig_vectors = np.load('knn_outputs/knn_vectors_'+image_name+'.npy')
    for n in range(len(cluster_numbers_list)):
        total_entropy = 0
        total_f_measure = 0
        cluster_numbers = cluster_numbers_list[n]
        # normalized_vector = normalize(eig_vectors[:, eig_vectors.shape[1]-cluster_numbers:eig_vectors.shape[1]])
        normalized_vector = normalize(eig_vectors[:, 0:cluster_numbers])
        ground_truth, img = load_image(image_name)
        assignments, centroids = k_means(normalized_vector, cluster_numbers, 1, 1000, cluster_numbers)
        data = get_image_data(image_name)
        data = data[0]
        clustered_image = np.zeros([data.shape[0], data.shape[1], 3], dtype=np.uint8)
        red_component = np.random.randint(256, size=cluster_numbers)
        blue_component = np.random.randint(256, size=cluster_numbers)
        green_component = np.random.randint(256, size=cluster_numbers)
        k = 0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                clustered_image[i][j] = [red_component[assignments[k]], green_component[assignments[k]],
                                         blue_component[assignments[k]]]

                k += 1
        # plt.imshow(clustered_image)
        # plt.show()
        plt.imsave('knn_outputs/' + image_name + '_' + str(cluster_numbers) + '.jpg',
                   clustered_image)
        for i in range(ground_truth.shape[1]):
            segment = ground_truth[0, i]
            segments_array = segment[0, 0][0]
            segments_array = np.ravel(segments_array)
            # print("================================")
            # print("number of clusters = ", cluster_numbers)
            # print("================================")
            # print("f1 measure : ", i)
            f_measure_val = f_measure(assignments, segments_array, cluster_numbers)
            # print("conditional entropy : ", i)
            conditional_entropy_val = conditional_entropy(assignments, segments_array, cluster_numbers)
            total_f_measure += f_measure_val
            total_entropy += conditional_entropy_val
        average_f_measure = total_f_measure / ground_truth.shape[1]
        average_entropy = total_entropy / ground_truth.shape[1]
        print("================================")
        print("image name = ", image_name)
        print("number of clusters = ", cluster_numbers)
        print("================================")
        print("average_entropy -> ", average_entropy)
        print("average_f_measure -> ", average_f_measure)
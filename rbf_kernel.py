from sklearn.metrics.pairwise import rbf_kernel as rbf
from scipy.sparse import csgraph
from sklearn.preprocessing import normalize
from helper import *
import matplotlib.pyplot as plt

cluster_numbers_list = [3, 5, 7, 9, 11]
# image_names_list = load()
image_names_list = ['183087']
gamma_list = [1, 10]
print(len(image_names_list))
x = 50
for z in range(len(image_names_list)):
    image_name = image_names_list[z]
    print("------------------------")
    print("image number = ", x)
    x += 1
    print("------------------------")
    image_data, img = get_image_data(image_name)
    reshaped_data = image_data.reshape(image_data.shape[0]*image_data.shape[1], image_data.shape[2])
    for p in range(len(gamma_list)):
        print("------------------------")
        print("gamma = ", gamma_list[p])
        print("------------------------")
        # RBF = rbf(reshaped_data, gamma=gamma_list[p])
        # # plt.matshow(RBF_01)
        # # plt.show()
        # lp_matrix = csgraph.laplacian(RBF, normed=False)
        # eigens = np.linalg.eigh(lp_matrix)
        # # eig_values = eigens[0]
        # eig_vectors = eigens[1]
        # # np.save('rbf_values_'+str(gamma_list[p])+"_"+image_name, eig_values)
        # np.save('rbf_outputs/rbf_vectors_'+str(gamma_list[p])+"_"+image_name, eig_vectors)
        eig_vectors = np.load('rbf_outputs/rbf_vectors_'+str(gamma_list[p])+'_'+image_name+'.npy')

        for n in range(len(cluster_numbers_list)):
            total_entropy = 0
            total_f_measure = 0
            cluster_numbers = cluster_numbers_list[n]
            # print("cluster number => ", cluster_numbers)
            # normalized_vector = normalize(eig_vectors[:, eig_vectors.shape[1]-cluster_numbers:eig_vectors.shape[1]])
            normalized_vector = normalize(eig_vectors[:, 0:cluster_numbers])
            ground_truth, img = load_image(image_name)
            # print(normalized_vector.shape)
            assignments, centroids = k_means(normalized_vector, cluster_numbers, 1, 1000, cluster_numbers)
            # print(centroids)
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
            plt.imsave('rbf_outputs/' + image_name + '_' + str(cluster_numbers) + '_'+str(gamma_list[p])+'.jpg', clustered_image)
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

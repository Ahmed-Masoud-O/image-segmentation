import numpy as np
from helper import *
import matplotlib.pyplot as plt
from helper import load_image
from helper import get_image_data

cluster_numbers_list = [5]
# image_names_list = load()
image_names_list = ['23080','24063','216053']
print(image_names_list)
with open("k_means_outputs/kMeans.txt", "w") as text_file:
    for z in range(len(image_names_list)):
        image_name = image_names_list[z]
        print("------------------------")
        print("image number = ", z)
        print("------------------------")
        for n in range(len(cluster_numbers_list)):
            total_entropy = 0
            total_f_measure = 0
            cluster_numbers = cluster_numbers_list[n]
            ground_truth, img = load_image(image_name)
            image_data, img = get_image_data(image_name)
            assignments, centroids = k_means(image_data, cluster_numbers)
            clustered_image = np.zeros([image_data.shape[0], image_data.shape[1], 3], dtype=np.uint8)
            red_component = np.random.randint(256, size=cluster_numbers)
            blue_component = np.random.randint(256, size=cluster_numbers)
            green_component = np.random.randint(256, size=cluster_numbers)
            k = 0
            for i in range(image_data.shape[0]):
                for j in range(image_data.shape[1]):
                    # clustered_image[i][j] = [red_component[assignments[k]], green_component[assignments[k]],
                    #                          blue_component[assignments[k]]]
                    clustered_image[i][j] = centroids[assignments[k]]
                    k += 1
            # plt.imshow(clustered_image)
            # plt.show()
            plt.imsave('k_means_outputs/'+image_name+'_'+str(cluster_numbers)+'.jpg', clustered_image)
            plt.imsave('k_means_outputs/' + image_name + '_' + '0' + '.jpg', img)
            # assignments =    [0,0,0,1,1,1,2,2,2,0]
            # segments_array = [1,2,1,2,3,1,3,2,1,1]

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
            print(f"================================\nimage name = {image_name}\nnumber of clusters = "
                  f"{cluster_numbers}\n================================\naverage_entropy -> {average_entropy}\n"
                  f"average_f_measure -> {average_f_measure}", file=text_file)




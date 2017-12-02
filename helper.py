from PIL import Image
import scipy.io
import numpy as np
import math
import os


def log2(x):
    return math.log(x) / math.log(2)


def load(my_folder='BSR/BSDS500/data/images/train'):
    names_list = []
    if os.path.isdir(my_folder):
        files = os.listdir(my_folder)
        for i in range(0, len(files)):
            if files[i][0:len(files[i])-4] == ".DS_S":
                continue
            names_list.append(files[i][0:len(files[i])-4])
    return names_list


def f_measure(assignments, segments_array, cluster_numbers):
    f_measures = list()
    unique, counts = np.unique(segments_array, return_counts=True)
    segment_array_count_dict = dict(np.asarray((unique, counts)).T)
    for k in range(cluster_numbers):
        precision_dict = {}
        count = 0
        if k not in assignments:
            continue
        for x in range(len(assignments)):
            if assignments[x] == k:
                count += 1
                if segments_array[x] in precision_dict:
                    precision_dict[segments_array[x]] = precision_dict[segments_array[x]] + 1
                else:
                    precision_dict[segments_array[x]] = 1
        key = max(precision_dict, key=precision_dict.get)
        value = precision_dict[max(precision_dict, key=precision_dict.get)]
        precision = value / count
        recall = value / segment_array_count_dict[key]
        f_measures.append(2 * precision * recall / (precision + recall))
    return np.asarray(f_measures).mean()


def conditional_entropy(assignments, segments_array, cluster_numbers):
    entropy_list = list()
    for k in range(cluster_numbers):
        precision_dict = {}
        count = 0
        entropy = 0
        if k not in assignments:
            continue
        for x in range(len(assignments)):
            if assignments[x] == k:
                count += 1
                if segments_array[x] in precision_dict:
                    precision_dict[segments_array[x]] = precision_dict[segments_array[x]] + 1
                else:
                    precision_dict[segments_array[x]] = 1
        # print("count: ", count)
        for key, value in precision_dict.items():
            entropy += -value / count * log2(value / count)
        entropy_list.append(entropy * count / len(assignments))
        # print(precision_dict)

    return sum(entropy_list)


def load_image(image_name='2092'):
    img = Image.open('BSR/BSDS500/data/images/train/'+image_name+'.jpg')
    mat = scipy.io.loadmat('BSR/BSDS500/data/groundTruth/train/'+image_name+'.mat')
    ground_truth = mat['groundTruth']
    return ground_truth, img


def get_image_data(img_name):
    img = Image.open('BSR/BSDS500/data/images/train/' + img_name + '.jpg')
    img.load()
    return np.asarray(img), img


def k_means(data, k=3, flag=0, max_iterations=1, dimensionality=3):
    # print(data.shape)
    n_row = data.shape[0]
    n_col = data.shape[1]
    center_row = np.random.choice(n_row, k, replace=False)
    center_col = np.random.choice(n_col, k, replace=False)
    centroids = np.zeros((k, dimensionality))
    centroids_old = np.zeros((k, dimensionality))

    for i in range(center_row.shape[0]):
        row = center_row[i]
        col = center_col[i]
        centroids[i] = data[row, col]
    if flag == 0:
        data_vector = data.reshape(data.shape[0] * data.shape[1], 3)
    else:
        data_vector = data
    x=0
    distances = [0]
    while (centroids_old != centroids).any() and x <= max_iterations:
        x += 1
        centroids_old = centroids.copy()
        distances = np.sqrt(((data_vector - centroids[:, np.newaxis])**2).sum(axis=2))
        cluster_assignments = np.argmin(distances, axis=0)
        for k in range(centroids.shape[0]):
            sum = np.zeros(dimensionality)
            count = 0
            for j in range(data_vector.shape[0]):
                if cluster_assignments[j] == k:
                    # print(data_vector[j])
                    sum += data_vector[j]
                    # print(sum)
                    count += 1
            if count > 0:
                centroids[k] = sum / count
        # centroids = np.array([data_vector[cluster_assignments == k].mean(axis=0) for k in range(centroids.shape[0])])
    # print(centroids)
    assignments = np.argmin(distances, axis=0)
    return assignments, centroids

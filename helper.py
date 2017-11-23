from PIL import Image
import scipy.io
import numpy as np


def load_image(image_name='2092'):
    img = Image.open('BSR/BSDS500/data/images/train/'+image_name+'.jpg')
    mat = scipy.io.loadmat('BSR/BSDS500/data/groundTruth/train/'+image_name+'.mat')
    ground_truth = mat['groundTruth']
    return ground_truth, img


def get_image_data(img_name):
    img = Image.open('BSR/BSDS500/data/images/train/' + img_name + '.jpg')
    img.load()
    return np.asarray(img), img


def k_means(data, k=3, flag=0, max_iterations=1000, dimensionality=3):
    print(data.shape)
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
    while (centroids_old != centroids).any() and x <= max_iterations:
        x += 1
        centroids_old = centroids.copy()
        distances = np.sqrt(((data_vector - centroids[:, np.newaxis])**2).sum(axis=2))
        cluster_assignments = np.argmin(distances, axis=0)
        centroids = np.array([data_vector[cluster_assignments == k].mean(axis=0) for k in range(centroids.shape[0])])
    # print(centroids)
    assignments = np.argmin(distances, axis=0)
    return assignments

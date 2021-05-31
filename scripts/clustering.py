import os
import pickle
import shutil
import time
from os import listdir
from os.path import isfile, join
from pathlib import Path
from shutil import copyfile

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pretty_print
import sklearn
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from sklearn import metrics
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

os.system("")
pp = pretty_print.Style()


def begin(selected_method, selected_model, selected_dataset):
    os.chdir(selected_dataset)

    pp.print_green('########## initializing model')
    model = initialize_model(selected_model)

    images = []

    pp.print_green('########## reading dataset')
    start_time = time.time()

    with os.scandir(selected_dataset) as files:
        for file in files:
            if file.name.endswith('.png') or file.name.endswith('.jpg'):
                images.append(file.name)

    pp.print_green('########## reading dataset done in {} seconds'.format(time.time() - start_time))

    data = {}

    pp.print_green('########## extracting features')
    start_time = time.time()

    for picture in tqdm(images):
        data[picture] = extract_features(picture, model)

    pp.print_green('########## extracting features done in {} seconds'.format(time.time() - start_time))

    filenames = np.array(list(data.keys()))

    features = np.array(list(data.values()))

    pp.print_green('########## reshaping')
    features = np.reshape(features, (features.shape[0], np.prod(features.shape[1:])))

    out_path = '../../results/{}/'.format(selected_model)
    pp.print_green('########## clustering')
    start_time = time.time()

    cluster_data(features, filenames, out_path, selected_dataset, selected_method, selected_model)

    pp.print_green('########## clustering done in {} seconds'.format(time.time() - start_time))


def initialize_model(model_type):
    if model_type == 'VGG16':
        model = VGG16(weights='imagenet', include_top=False)
    elif model_type == 'ResNet50':
        model = ResNet50(weights='imagenet', include_top=False)
    else:
        model = InceptionV3(weights='imagenet', include_top=False)

    return model


def cluster_data(feat, filenames, results_path, input_path, user_model, selected_model):
    if user_model == 'DBSCAN':
        eps = float(input('max distance to the neighbor: '))
        min_samples = int(input('number of points to be considered core point:'))
        method = DBSCAN(eps=eps, min_samples=min_samples, algorithm='auto', metric='minkowski', p=2, n_jobs=-1)
    elif user_model == 'k-means':
        n_clusters = int(input('number of clusters: '))
        random_state = int(input('random state: '))
        method = KMeans(n_clusters=n_clusters, n_jobs=-1, random_state=random_state)
    elif user_model == 'knn':
        method = KNeighborsClassifier(n_neighbors=5)
    else:
        method = OPTICS(min_samples=5, n_jobs=-1)

    if user_model == 'knn':
        # method.fit(feat, dane_treningowe)
        pass
    else:
        method.fit(feat)

    groups = {}
    for file, cluster in zip(filenames, method.labels_):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)

        try:
            copyfile(os.path.abspath('{}/{}'.format(input_path, file)),
                     os.path.abspath('{}/{}/{}'.format(results_path, cluster, file)))
        except FileNotFoundError:
            os.mkdir(os.path.abspath('{}/{}/'.format(results_path, cluster)))
            copyfile(os.path.abspath('{}/{}'.format(input_path, file)),
                     os.path.abspath('{}/{}/{}'.format(results_path, cluster, file)))

    # Evaluation
    number_of_clusters = len(set(method.labels_))
    if number_of_clusters > 1:
        clustering_evaluation(feat, method.labels_, number_of_clusters, selected_model, user_model)
        silhouette_for_every_sample(feat, method.labels_, number_of_clusters)


def extract_features(file, model):
    img = load_img(file, target_size=(224, 224))
    img = np.array(img)
    reshaped_img = img.reshape(1, 224, 224, 3)
    imgx = preprocess_input(reshaped_img)
    features = model.predict(imgx, use_multiprocessing=True)

    return features


def clustering_evaluation(features, labels, number_of_clusters, model, clustering_method):
    # ==================================== Silhouette ===============================
    silhouette_avg = silhouette_score(features, labels)
    print("Number of clusters: ", number_of_clusters, " model: ", model,
          " clustering method: ", clustering_method, ", the average silhouette_score is: ", silhouette_avg)

    # ================================ Calinski-Harabasz ============================
    calinski_harabasz = metrics.calinski_harabasz_score(features, labels)
    print("Number of clusters: ", number_of_clusters, " model: ", model,
          " clustering method: ", clustering_method, ", the Calinski-Harabasz score is: ", calinski_harabasz)

    # ================================ Davies-Bouldin ===============================
    davies_bouldin = metrics.davies_bouldin_score(features, labels)
    print("Number of clusters: ", number_of_clusters, " model: ", model,
          " clustering method: ", clustering_method, ", the Davies-Bouldin score is: ", davies_bouldin)


def silhouette_for_every_sample(features, labels, number_of_clusters):
    silhouette_avg = silhouette_score(features, labels)
    # ================== Compute the silhouette scores for each sample ==============
    sample_silhouette_values = silhouette_samples(features, labels)
    y_lower = 10
    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)

    for i in range(number_of_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / number_of_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.show()

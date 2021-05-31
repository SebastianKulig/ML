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
from sklearn.decomposition import PCA
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
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize
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
    pp.print_red("########## {} {} {}".format(selected_method, selected_model, selected_dataset))
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
    features_dimension = features.ndim

    pp.print_green('########## reshaping')
    features = np.reshape(features, (features.shape[0], np.prod(features.shape[1:])))

    pp.print_green('########## normalizing')
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features = normalize(features)

    pp.print_green('########## pca to 95% variance')
    pca = PCA(n_components=0.95, svd_solver='full')
    pca.fit(features)
    features = pca.transform(features)
    pp.print_green('########## pca to 95% variance done')

    out_path = '../../results/{}/'.format(selected_model)
    pp.print_green('########## clustering')
    start_time = time.time()

    cluster_data(features, filenames, out_path, selected_dataset, selected_method, features_dimension)

    pp.print_green('########## clustering done in {} seconds'.format(time.time() - start_time))


def initialize_model(model_type):
    if model_type == 'VGG16':
        model = VGG16(weights='imagenet', include_top=False)
    elif model_type == 'Resnet50':
        model = ResNet50(weights='imagenet', include_top=False)
    else:
        model = InceptionV3(weights='imagenet', include_top=False)

    return model


def cluster_data(feat, filenames, results_path, input_path, user_model, dims):
    if user_model == 'DBSCAN':
        pp.print_green('########## calculating neighbors')
        eps = neighbors_plot(feat)
        method = DBSCAN(eps=eps, min_samples=(dims * 2), algorithm='auto', metric='minkowski', p=2, n_jobs=-1)
    elif user_model == 'k-means':
        n_clusters = int(input('number of clusters: '))
        random_state = int(input('random state: '))
        method = KMeans(n_clusters=n_clusters, n_jobs=-1, random_state=random_state)
    elif user_model == 'knn':
        method = KNeighborsClassifier(n_neighbors=5)
    else:
        final_xi = 0.004
        method = OPTICS(min_samples=10, n_jobs=-1, cluster_method='xi', xi=final_xi)

    if user_model == 'knn':
        # method.fit(feat, dane_treningowe)
        pass
    else:
        pp.print_green('########## .fit(data)')
        method.fit(feat)
        if user_model == 'DBSCAN':
            plot_dbscan_clusters(feat, method)
        if user_model == 'OPTICS':
            make_reachability_plot(method, len(feat), 10)

    pp.print_green('########## saving results to {}'.format(os.path.abspath('{}/'.format(results_path))))
    # groups = {}
    # for file, cluster in zip(filenames, method.labels_):
    #     if cluster not in groups.keys():
    #         groups[cluster] = []
    #         groups[cluster].append(file)
    #     else:
    #         groups[cluster].append(file)
    #
    #     try:
    #         copyfile(os.path.abspath('{}/{}'.format(input_path, file)),
    #                  os.path.abspath('{}/{}/{}'.format(results_path, cluster, file)))
    #     except FileNotFoundError:
    #         os.mkdir(os.path.abspath('{}/{}/'.format(results_path, cluster)))
    #         copyfile(os.path.abspath('{}/{}'.format(input_path, file)),
    #                  os.path.abspath('{}/{}/{}'.format(results_path, cluster, file)))
    
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


def neighbors_plot(feat):
    neighbors_method = NearestNeighbors(n_neighbors=2, metric='minkowski', p=2, n_jobs=-1)
    neighbors = neighbors_method.fit(feat)
    distances, indices = neighbors.kneighbors(feat)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    derivative_dist = np.gradient(distances) / (1/len(distances))
    half_index = int(len(derivative_dist)/2)
    border = np.percentile(derivative_dist, 95)
    item = np.where(derivative_dist > border)[0]
    try:
        final_index = item[np.where(item > half_index)][0]
    except IndexError:
        border = np.percentile(derivative_dist, 90)
        item = np.where(derivative_dist > border)[0]
        final_index = item[np.where(item > half_index)][0]
    plt.axhline(y=distances[final_index], color='r', linestyle='--')
    plt.plot(distances)
    plt.text(0, distances[final_index], str(distances[final_index]))
    plt.xlabel('sample number')
    plt.ylabel('distance to the neighbor')
    plt.title('sorted distances between points')
    plt.show()

    return distances[final_index]

def plot_dbscan_clusters(data, model):

    pca = PCA(n_components=2)
    pca.fit(data)
    data_PCA = pca.transform(data)

    labels = model.labels_

    n_clusters_ = len(labels) - (1 if -1 in labels else 0)
    unique_labels = set(labels)
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[model.core_sample_indices_] = True
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = data_PCA[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], '*', color=tuple(col))

        xy = data_PCA[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], '.', color=tuple(col))

    plt.title('Clusters, star means core point, black = not clustered')
    plt.show()


def make_reachability_plot(model, data_array_length, dimension):

    space = np.arange(data_array_length)
    reachability = model.reachability_[model.ordering_]
    labels = model.labels_[model.ordering_]

    klas = np.unique(labels).tolist()

    for klass in klas:
        Xk = space[labels == klass]
        Rk = reachability[labels == klass]
        if klass == -1:
            plt.plot(Xk, Rk, 'k.')
        else:
            plt.plot(Xk, Rk, '.')

    plt.title('Reachability Plot, min_sample = {}, xi = 0.004, -1 means not clustered'.format(dimension))
    plt.ylabel('Reachability (epsilon distance)')
    plt.legend(klas)
    plt.show()


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

if __name__ =='__main__':
    exact_path = '..\\datasets\\geo_shapes\\'
    exact_path = os.path.abspath(exact_path)
    begin('OPTICS', 'ResNet50', exact_path)
    
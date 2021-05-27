import os
from shutil import copyfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.neighbors import KNeighborsClassifier

from keras.preprocessing.image import load_img
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import preprocess_input
from keras.applications.inception_v3 import InceptionV3

from keras.preprocessing.image import img_to_array
from keras.models import Model
from keras.preprocessing import image
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import sklearn
import shutil
from os import listdir
from os.path import isfile, join
from keras.models import Model
import pickle


def begin(selected_method, selected_model, selected_dataset):
    os.chdir(selected_dataset)

    images = []

    if selected_model == 'VGG16':
        model = VGG16(weights='imagenet', include_top=False)
    elif selected_model == 'Resnet50':
        model = ResNet50(weights='imagenet', include_top=False)
    else:
        model = InceptionV3(weights='imagenet', include_top=False)

    # creates a ScandirIterator aliased as files
    with os.scandir(selected_dataset) as files:
        # loops through each file in the directory
        for file in files:
            if file.name.endswith('.png') or file.name.endswith('.jpg'):
                # adds only the image files to the images list
                images.append(file.name)

    print("finished dataset reading")

    data = {}

    print("extracting features")
    for flower in images:
        # try to extract the features and update the dictionary
        feat = extract_features(flower, model)
        data[flower] = feat

    print("extracting features done")

    # get a list of the filenames
    filenames = np.array(list(data.keys()))

    feat = np.array(list(data.values()))

    # get a list of just the features
    feat = np.reshape(feat, (feat.shape[0], np.prod(feat.shape[1:])))
    print("reshape done")

    if selected_method == 'DBSCAN':
        image_path = '../../'
        image_path = os.path.abspath(image_path)
        neighbors_plot(feat, image_path)

    out_path = '../../results/{}/'.format(selected_model)
    out_path = os.path.abspath(out_path)

    print('clustering data')
    cluster_data(feat, filenames, out_path, selected_dataset, selected_method)


def cluster_data(feat, filenames, out_path, path, user_model):
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
    # images
    groups = {}
    for file, cluster in zip(filenames, method.labels_):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(file)
        else:
            groups[cluster].append(file)
        try:
            copyfile(path + Path("/") + str(file), out_path + Path("/") + str(cluster) + Path("/") + str(file))
        except:
            os.mkdir(out_path + Path("/") + str(cluster))
            copyfile(path + Path("/") + str(file), out_path + Path("/") + str(cluster) + Path("/") + str(file))


def neighbors_plot(feat, out_path):
    print('calculating distances')
    neigh = NearestNeighbors(n_neighbors=2)
    neighbors = neigh.fit(feat)
    distances, indices = neighbors.kneighbors(feat)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.plot(distances)
    plt.xlabel('sample number')
    plt.ylabel('distance to the neighbor')
    plt.title('sorted distances between points')
    plt.savefig(out_path + Path("/") + 'neighbor.png')
    plt.show()
    print('plot saved to {}'.format(out_path + Path("/") + 'neighbor.png'))


def extract_features(file, model):
    # load the image as a 224x224 array
    img = load_img(file, target_size=(224, 224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img)
    # reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels)
    reshaped_img = img.reshape(1, 224, 224, 3)
    # prepare image for model
    imgx = preprocess_input(reshaped_img)
    # get the feature vector
    features = model.predict(imgx, use_multiprocessing=True)
    return features

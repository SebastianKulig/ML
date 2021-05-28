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
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors

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

    for picture in images:
        data[picture] = extract_features(picture, model)

    pp.print_green('########## extracting features done in {} seconds'.format(time.time() - start_time))

    filenames = np.array(list(data.keys()))

    features = np.array(list(data.values()))

    pp.print_green('########## reshaping')
    features = np.reshape(features, (features.shape[0], np.prod(features.shape[1:])))

    out_path = '../../results/{}/'.format(selected_model)
    pp.print_green('########## clustering')
    start_time = time.time()

    cluster_data(features, filenames, out_path, selected_dataset, selected_method)

    pp.print_green('########## clustering done in {} seconds'.format(time.time() - start_time))


def initialize_model(model_type):
    if model_type == 'VGG16':
        model = VGG16(weights='imagenet', include_top=False)
    elif model_type == 'Resnet50':
        model = ResNet50(weights='imagenet', include_top=False)
    else:
        model = InceptionV3(weights='imagenet', include_top=False)

    return model


def cluster_data(feat, filenames, results_path, input_path, user_model):
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


def extract_features(file, model):
    img = load_img(file, target_size=(224, 224))
    img = np.array(img)
    reshaped_img = img.reshape(1, 224, 224, 3)
    imgx = preprocess_input(reshaped_img)
    features = model.predict(imgx, use_multiprocessing=True)

    return features

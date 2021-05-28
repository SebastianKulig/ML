import os
import sys
from shutil import copyfile
import numpy as np
import time

import pretty_print

from keras.preprocessing.image import load_img
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import preprocess_input
from keras.applications.inception_v3 import InceptionV3

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


def cluster_data(feat, filenames, results_path, input_path, user_model):
    if user_model == 'DBSCAN':
        # TODO DBSCAN that returns clustering method object used to call method.fit(images)
        method = 1
    elif user_model == 'k-means':
        # TODO k-means that returns clustering method object used to call method.fit(images)
        method = 1
    elif user_model == 'knn':
        # TODO knn that returns clustering method object used to call method.fit(images)
        method = 1
    else:
        # TODO OPTICS that returns clustering method object used to call method.fit(images)
        method = 1

    if method == 1:
        sys.exit()

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


def initialize_model(model_type):
    if model_type == 'VGG16':
        model = VGG16(weights='imagenet', include_top=False)
    elif model_type == 'Resnet50':
        model = ResNet50(weights='imagenet', include_top=False)
    else:
        model = InceptionV3(weights='imagenet', include_top=False)

    return model

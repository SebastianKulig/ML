import os
from pathlib import Path

import inquirer

import clustering


def user_input():
    questions = [
        inquirer.List('cluster_method',
                      message='What clustering method would you like to use?',
                      choices=['k-means', 'knn', 'DBSCAN', 'OPTICS'],
                      ),

        inquirer.List('dataset',
                      message='What dataset would you like to use?',
                      choices=['Flowers - 5 classes, ~3500 images',
                               'Geometric shapes - 9 classes, 90000 images',
                               'Dog Breeds - 120 classes, >20000 images',
                               'Natural Images Dataset - 8 classes, ~7000 images'],
                      ),

        inquirer.List('model',
                      message='What model would you like to use?',
                      choices=['VGG16',
                               'ResNet50',
                               'InceptionV3'],
                      ),
    ]
    answers = inquirer.prompt(questions)
    return answers


def translate_dataset_path(user_friendly_name):
    if user_friendly_name == 'Flowers - 5 classes, ~3500 images':
        exact_path = Path('../datasets/flowers/')
    elif user_friendly_name == 'Geometric shapes - 9 classes, 90000 images':
        exact_path = Path('../datasets/geo_shapes/')
    elif user_friendly_name == 'Dog Breeds - 120 classes, >20000 images':
        exact_path = Path('../datasets/dog_breeds/')
    else:
        exact_path = Path('../datasets/natural_images/')

    exact_path = os.path.abspath(exact_path)
    return exact_path


def main():
    user_choices = user_input()
    dataset_path = translate_dataset_path(user_choices['dataset'])
    clustering.begin(user_choices['cluster_method'], user_choices['model'], dataset_path)


if __name__ == '__main__':
    main()

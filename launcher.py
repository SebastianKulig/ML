import inquirer


def user_input():
    questions = [
        inquirer.List('cluster_method',
                      message='What clustering method would you like to use?',
                      choices=['k-means', 'knn', 'DBSCAN', 'other'],
                      ),

        inquirer.List('dataset',
                      message='What dataset would you like to use?',
                      choices=['flowers - 3500 images',
                               'geometric figures - 9000 images',
                               'DogBreeds - 120 classes',
                               'Seba\'s cool dataset'],
                      ),

        inquirer.List('model',
                      message='What model would you like to use?',
                      choices=['VGG16',
                               'Resnet50',
                               'InceptionV3'],
                      ),
    ]
    answers = inquirer.prompt(questions)
    return answers


def translate_dataset_path(user_friendly_name):
    if user_friendly_name == 'flowers - 3500 images':
        exact_path = r'../datasets/flowers/'
    elif user_friendly_name == 'geometric figures - 9000 images':
        exact_path = r'../datasets/geo_shapes/'
    elif user_friendly_name == 'DogBreeds - 120 classes':
        exact_path = r'../datasets/dogs/'
    else:
        exact_path = r'../datasets/cool_ones/'

    return exact_path


def launch_clustering(cluster_type, path, model):
    print('chosen clustering method: {}\nchosen model: {}\ndataset location: {}'.format(cluster_type,
                                                                                        model,
                                                                                        path))


def main():
    user_choices = user_input()
    dataset_path = translate_dataset_path(user_choices['dataset'])
    launch_clustering(user_choices['cluster_method'], dataset_path, user_choices['model'])


if __name__ == '__main__':
    main()

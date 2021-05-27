import inquirer
import clustering
import os


def user_input():
    questions = [
        inquirer.List('local_os',
                      message='What OS are you using?',
                      choices=['Unix', 'Windows'],
                      ),

        inquirer.List('cluster_method',
                      message='What clustering method would you like to use?',
                      choices=['k-means', 'knn', 'DBSCAN', 'OPTICS'],
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


def translate_dataset_path(user_friendly_name, local_os):
    slash = '/' if local_os == 'Unix' else '\\'
    if user_friendly_name == 'flowers - 3500 images':
        exact_path = '..%sdatasets%sflowers%s' % (slash, slash, slash)
    elif user_friendly_name == 'geometric figures - 9000 images':
        exact_path = '..%sdatasets%sgeo_shapes%s' % (slash, slash, slash)
    elif user_friendly_name == 'DogBreeds - 120 classes':
        exact_path = '..%sdatasets%sdogs%s' % (slash, slash, slash)
    else:
        exact_path = '..%sdatasets%scool_ones%s' % (slash, slash, slash)

    exact_path = os.path.abspath(exact_path)
    return exact_path


def launch_clustering(cluster_type, path, model):
    print('chosen clustering method: {}\nchosen model: {}\ndataset location: {}'.format(cluster_type,
                                                                                        model,
                                                                                        path))


def main():
    user_choices = user_input()
    dataset_path = translate_dataset_path(user_choices['dataset'], user_choices['local_os'])
    clustering.begin(user_choices['cluster_method'], user_choices['model'], dataset_path)


if __name__ == '__main__':
    main()

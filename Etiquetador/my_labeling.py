__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

import numpy as np

from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_retrieval
from KNN import __authors__, __group__, KNN
from Kmeans import __authors__, __group__, KMeans, distance, get_colors


def retrieval_by_color(images, kmeans_tags, color):
    matching_images = []

    for x, tags in enumerate(kmeans_tags):
        if all(tag in tags for tag in color):
            matching_images.append(images[x])
    visualize_retrieval(matching_images, 10)


def retrieval_by_shape(images, knn_tags, shape):
    matching_images = []

    for i, tags in enumerate(knn_tags):
        if shape in tags:
            matching_images.append(images[i])
    visualize_retrieval(matching_images, 10)


def retrieval_combined(images, shape_tags, color_tags, shape, color):
    matching_images = []
    i = 0
    for shape_tags, color_tags in zip(shape_tags, color_tags):
        if shape == shape_tags and color in color_tags:
            matching_images.append(images[i])
        i += 1
    if not matching_images:
        return "Nothing found"
    else:
        visualize_retrieval(matching_images, 10)


def get_shape_accuracy(tags, ground_truth):
    if len(tags) != len(ground_truth):
        raise ValueError("Length of 'tags' and 'ground_truth' must be the same.")

    total = len(tags)
    correct = sum(1 for prediction, truth in zip(tags, ground_truth) if prediction == truth)

    return (correct / total) * 100 if total > 0 else 0


if __name__ == '__main__':
    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    # You can start coding your functions here

    """
    Tests qualitativos
    """
    #retrieval_by_color(test_imgs, test_color_labels, ['Black', 'Blue'])
    #retrieval_by_shape(test_imgs, test_class_labels, 'Jeans')
    #retrieval_by_shape(test_imgs, test_class_labels, 'Shorts')
    #retrieval_by_shape(test_imgs, test_class_labels, 'Heels')
    #retrieval_combined(test_imgs, test_class_labels, test_color_labels, 'Flip Flops', ['Blue'])

    """
        Tests quantitativos
    """
    kn = KNN(train_imgs, train_class_labels)
    kn.predict(test_imgs, 60)

    shape_percent = get_shape_accuracy(kn.get_class(), test_class_labels)
    print("Percentatge: ", round(shape_percent, 2), "%")

    # prueba retrieval_by_color

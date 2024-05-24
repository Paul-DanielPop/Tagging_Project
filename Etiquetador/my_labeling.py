__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

import time
import numpy as np
import utils
import matplotlib.pyplot as plt

#from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_retrieval
from utils_data import *
from KNN import __authors__, __group__, KNN
from Kmeans import __authors__, __group__, KMeans, distance, get_colors
from PIL import Image


def retrieval_by_color(images, kmeans_tags, color):
    matching_images = []
    msg_colors = ", ".join(color)
    print(f"Searching for {msg_colors} images")
    for x, tags in enumerate(kmeans_tags):
        if all(tag in tags for tag in color):
            matching_images.append(images[x])
    visualize_retrieval(matching_images, 10, title=f'{color}')


def retrieval_by_shape(images, knn_tags, shape):
    matching_images = []
    print(f"Searching for {shape}")
    for i, tags in enumerate(knn_tags):
        if shape in tags:
            matching_images.append(images[i])
    visualize_retrieval(matching_images, 10, title=f'{shape}')


def retrieval_combined(images, shape_tags, color_tags, shape, color):
    matching_images = []
    msg_colors = ", ".join(color)
    print(f"Searching for {msg_colors} {shape[0]}")
    for i in range(len(images)):
        shape_match = all(sh in shape_tags[i] for sh in shape)
        color_match = all(col in color_tags[i] for col in color)

        if shape_match and color_match:
            matching_images.append(images[i])

    visualize_retrieval(matching_images, 10, title=f'{color}{shape}')


def get_color_accuracy(kmeans_tags, ground_truth_tags):
    correct_predictions = 0
    total_length = 0

    for kmeans_list, gt_list in zip(kmeans_tags, ground_truth_tags):
        for label in kmeans_list:
            if label in gt_list:
                correct_predictions += 1
            total_length += 1

    color_accuracy = (correct_predictions / total_length) * 100
    return color_accuracy


def get_shape_accuracy(tags, ground_truth):
    if len(tags) != len(ground_truth):
        raise ValueError("Length of 'tags' and 'ground_truth' must be the same.")

    total = len(tags)
    correct = sum(1 for prediction, truth in zip(tags, ground_truth) if prediction == truth)

    return (correct / total) * 100 if total > 0 else 0


def kmeans_statistics(train_images, train_class_gt, images_to_classify, color_gt, class_gt, kmax, show_graph=False,
                      show_image=False, view_statistics=False, options=None):
    global_statistics = []

    knn = KNN(train_images, train_class_gt)
    result_shape_labels = knn.predict(imgs, 5)

    for i, image in enumerate(images_to_classify):
        statistics = []

        for k in range(2, kmax + 1):
            start_time = time.time()
            kmeans = KMeans(image, k, options)
            kmeans.fit()
            total_time = time.time() - start_time
            wcd = kmeans.withinClassDistance()
            n_iter = kmeans.num_iter
            title = f"K={k}"
            if show_graph:
                Plot3DCloud(kmeans, 1, kmax - 1, k - 1, title)

            colors = get_colors(kmeans.centroids)
            statistic = {
                'K': k,
                'WCD': wcd,
                'Num_iterations': n_iter,
                'Convergence_time': total_time,
                'Found_color': set(colors),
                'Color_gt': color_gt[i],
                'Color_accuracy': get_color_accuracy([list(set(colors))], [color_gt[i]]),
                'Found_shape': result_shape_labels[i],
                'Shape_gt': class_gt[i],
                'Shape_accuracy': get_shape_accuracy([result_shape_labels[i]], [class_gt[i]])
            }

            statistics.append(statistic)
            print_statistics(statistic)

        if show_graph:
            plt.show()
        if show_image:
            computed_image = Image.fromarray(image)
            computed_image.show()

        global_statistics.append(statistics)

    if view_statistics:
        visualize_statistics(global_statistics)


def test_retrieval_by_color(images, gt):
    result_color_labels = []

    for image in images:
        km = KMeans(image, 5)
        km.fit()
        colors = list(set(get_colors(km.centroids)))
        result_color_labels.append(colors)

    accuracy = get_color_accuracy(result_color_labels, gt)
    print(f'Color accuracy: {accuracy}')

    # Test 1
    retrieval_by_color(images, result_color_labels, ['Black'])

    # Test 2
    retrieval_by_color(images, result_color_labels, ['Pink', 'Grey'])

    # Test 3
    retrieval_by_color(images, result_color_labels, ['White', 'Black', 'Grey'])

    # Test 4
    retrieval_by_color(images, result_color_labels, ['Brown', 'Grey', 'Orange', 'White'])

    # Test 5
    retrieval_by_color(images, result_color_labels, ['White', 'Orange', 'Purple', 'Pink', 'Black'])


def test_retrieval_by_shape(images, shape_gt):
    knn = KNN(images, shape_gt)
    result_shape_labels = knn.predict(images, 10)

    shape_acc = get_shape_accuracy(result_shape_labels, shape_gt)
    print(f'Shape accuracy: {shape_acc}')

    # Test 1
    retrieval_by_shape(images, result_shape_labels, 'Jeans')

    # Test 2
    retrieval_by_shape(images, result_shape_labels, 'Shorts')

    # Test 3
    retrieval_by_shape(images, result_shape_labels, 'Dresses')

    # Test 4
    retrieval_by_shape(images, result_shape_labels, 'Shirts')

    # Test 5
    retrieval_by_shape(images, result_shape_labels, 'Flip Flops')


def test_retrieval_combined(images, color_gt, shape_gt):
    knn = KNN(images, shape_gt)
    result_color_labels = []
    result_shape_labels = knn.predict(images, 10)
    for image in images:
        km = KMeans(image, 5)
        km.fit()
        colors = get_colors(km.centroids)
        result_color_labels.append(colors)

    # Test 1
    retrieval_combined(images, result_shape_labels, result_color_labels, ['Shorts'], ['Blue'])

    # Test 2
    retrieval_combined(images, result_shape_labels, result_color_labels, ['Dresses'], ['Black'])

    # Test 3
    retrieval_combined(images, result_shape_labels, result_color_labels, ['Sandals'], ['Yellow', 'Brown'])

    # Test 4
    retrieval_combined(images, result_shape_labels, result_color_labels, ['Socks'], ['Black', 'Orange', 'White'])


def test_kmeans_statistics():
    images_to_classify = cropped_images[:2]
    kmeans_statistics(train_imgs, train_class_labels, images_to_classify,
                      color_labels, class_labels, 5, True, True, True)

    opt = {
        'km_init': 'kmeans++'
    }

    images_to_classify = cropped_images[:2]
    kmeans_statistics(train_imgs, train_class_labels, images_to_classify,
                      color_labels, class_labels, 5, True, True, True, options=opt)

    opt = {
        'km_init': 'random'
    }

    images_to_classify = cropped_images[:2]
    kmeans_statistics(train_imgs, train_class_labels, images_to_classify,
                      color_labels, class_labels, 5, True, True, True, options=opt)


def visualize_statistics(statistics):
    fig, axs = plt.subplots(1, 3, figsize=(16, 5))

    num_images = len(statistics)
    colors = plt.cm.get_cmap('tab10', num_images)

    for idx, image_stats in enumerate(statistics):
        # Extraer los valores de las estadísticas para cada K
        Ks = [stat['K'] for stat in image_stats]
        WCDs = [stat['WCD'] for stat in image_stats]
        convergence_times = [stat['Convergence_time'] for stat in image_stats]
        color_accuracy = [stat['Color_accuracy'] for stat in image_stats]

        # Gráfico de WCD vs K
        axs[0].plot(Ks, WCDs, marker='o', label=f'Image {idx + 1}', color=colors(idx))
        axs[0].set_title('Within-Class-Distance (WCD) vs K', fontsize=10)
        axs[0].set_xlabel('Number of Clusters (K)', fontsize=8)
        axs[0].set_ylabel('WCD', fontsize=8)
        axs[0].tick_params(axis='both', which='major', labelsize=8)
        axs[0].grid(True)

        # Gráfico de tiempo de convergencia vs K
        axs[1].plot(Ks, convergence_times, marker='o', label=f'Image {idx + 1}', color=colors(idx))
        axs[1].set_title('Convergence Time vs K', fontsize=10)
        axs[1].set_xlabel('Number of Clusters (K)', fontsize=8)
        axs[1].set_ylabel('Convergence Time (seconds)', fontsize=8)
        axs[1].tick_params(axis='both', which='major', labelsize=8)
        axs[1].grid(True)

        axs[2].plot(Ks, color_accuracy, marker='o', label=f'Image {idx + 1}', color=colors(idx))
        axs[2].set_title('Accuracy vs K', fontsize=10)
        axs[2].set_xlabel('K', fontsize=10)
        axs[2].set_ylabel('Accuracy (%)', fontsize=10)
        axs[2].tick_params(axis='both', which='major', labelsize=8)
        axs[2].grid(True)

    # Añadir leyenda a cada gráfico
    for ax in axs:
        ax.legend()

    # Ajustar espacio entre gráficos
    plt.tight_layout(pad=4.0)

    # Mostrar los gráficos
    plt.show()


def print_statistics(statistic):
    for key, value in statistic.items():
        print(f'{key}: {value}')
    print()


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

    """Tests retrieval_by_color"""
    # test_retrieval_by_color(train_imgs[:1000], train_color_labels[:1000])

    """Tests retrieval_by_shape"""
    # test_retrieval_by_shape(train_imgs[:1000], train_class_labels[:1000])

    """Tests retrieval_combined"""
    test_retrieval_combined(train_imgs[:300], train_color_labels[:300], train_class_labels[:300])

    """Tests kmeans_statistics"""
    test_kmeans_statistics()
    """
        Tests quantitativos
    """
    
    kn = KNN(train_imgs, train_class_labels)
    kn.predict(test_imgs, 60)

    shape_percent = get_shape_accuracy(kn.get_class(), test_class_labels)
    print("Percentatge: ", round(shape_percent, 2), "%")
    
    #Test Find_BestK
    test_imgs = cropped_images[:10]
    knn = KNN(imgs, class_labels)
    resultats_forma = knn.predict(imgs, 10)
    for i,image in enumerate(test_imgs):
        #kmeans = KMeans(image, 3,options)
        kmeans = KMeans(image, 3)
        kmeans.find_bestK(10)
        colors = []
        for k in range(kmeans.K):
            # Generar un color basat en el valor de k
            colors.append(list(kmeans.centroids[k]))
        color = get_colors(np.array(colors))
        color=set(color)
        imageObj = Image.fromarray(image)
        imageObj.show()
        Plot3DCloud(kmeans,1,1,1)
        plt.show()
        print("K:", kmeans.K)
        print("WCD:", kmeans.WCD)
        print("Iteracions:", kmeans.num_iter)
        print("Color Trobat:", color)
        print("Color Predit:", color_labels[i])
        print("Forma Trobat:", resultats_forma[i][0])
        print("Forma Predit:", class_labels[i])
    
    # test_kmeans_statistics()


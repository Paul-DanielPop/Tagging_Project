__authors__ = 'TO_BE_FILLED'
__group__ = 'TO_BE_FILLED'

import random
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


def kmeans_knn_statistics(knn_train_images, knn_train_gt, test_images, num_images, test_color_gt, test_class_gt, kmax,
                          kmeans_options=None, show_graph=False, show_image=False, view_statistics=False):
    global_statistics = []
    knn = KNN(knn_train_images, knn_train_gt)
    result_shape_labels = knn.predict(imgs, 5)
    random_indices = random.sample(range(0, len(test_images) - 1), num_images)
    for i in random_indices:
        statistics = []

        image = test_images[i]

        for k in range(2, kmax + 1):
            start_time = time.time()
            kmeans = KMeans(image, k, kmeans_options)
            kmeans.fit()
            total_time = time.time() - start_time
            wcd = kmeans.withinClassDistance()
            n_iter = kmeans.num_iter
            title = f"K={k}"
            if show_graph:
                MyPlot3DCloud(kmeans, 1, kmax - 1, k - 1, title)

            colors = get_colors(kmeans.centroids)
            statistic = {
                'K': k,
                'init_method': kmeans_options['km_init'],
                'WCD': wcd,
                'Num_iterations': n_iter,
                'Convergence_time': total_time,
                'Found_color': set(colors),
                'Color_gt': test_color_gt[i],
                'Color_accuracy': get_color_accuracy([list(set(colors))], [test_color_gt[i]]),
                'Found_shape': result_shape_labels[i],
                'Shape_gt': test_class_gt[i],
                'Shape_accuracy': get_shape_accuracy([result_shape_labels[i]], [test_class_gt[i]])
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


def kmeans_statistics_nonRandom(train_images, train_class_gt, images_to_classify, color_gt, class_gt, kmax, show_graph=False,
                      show_image=False, view_statistics=False, options=None):
    global_statistics = []
    knn = KNN(train_images, train_class_gt)
    result_shape_labels = knn.predict(images_to_classify, 5)

    for i, image in enumerate(images_to_classify):
        image = images_to_classify[i]
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
                MyPlot3DCloud(kmeans, 1, kmax - 1, k - 1, title)
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
        
def kmeans_statistics_nonRandom_Specific(train_images, train_class_gt, images_to_classify, color_gt, class_gt, kmax, show_graph=False,
                      show_image=False, view_statistics=False, options=None):
    global_statistics = []
    knn = KNN(train_images, train_class_gt)
    result_shape_labels = knn.predict(train_images, 5)

    for i, image in enumerate(images_to_classify):
        image = images_to_classify[i]
        statistics = []
        hac = 0
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
            if (hac <= get_color_accuracy([list(set(colors))], [color_gt[i]])):
                hac = get_color_accuracy([list(set(colors))], [color_gt[i]])
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

def kmeans_statistics_nonRandom_plusF(train_images, train_class_gt, images_to_classify, color_gt, class_gt, kmax, show_graph=False,
                      show_image=False, view_statistics=False, options=None):
    global_statistics = []
    knn = KNN(train_images, train_class_gt)
    result_shape_labels = knn.predict(imgs, 5)

    for i, image in enumerate(images_to_classify):
        image = images_to_classify[i]
        statistics = []

        for k in range(2, kmax + 1):
            start_time = time.time()
            kmeans = KMeans(image, k, options)
            kmeans.fit()
            total_time = time.time() - start_time
            wcd = kmeans.withinClassDistance()
            icd = kmeans.inter_class_distance()
            fisher = kmeans.Fisher_coefficient()
            n_iter = kmeans.num_iter
            title = f"K={k}"
            if show_graph:
                MyPlot3DCloud(kmeans, 1, kmax - 1, k - 1, title)
            colors = get_colors(kmeans.centroids)
            statistic = {
                'K': k,
                'WCD': wcd,
                'ICD': icd,
                'FISHER': fisher,
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
        visualize_statistics_fisher(global_statistics)


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
    retrieval_combined(images, result_shape_labels, result_color_labels, ['Shirts'], ['White', 'Orange'])


def test_kmeans_statistics():
    #images_to_classify = cropped_images[:10]
    opt = {
        'km_init': 'first'
    }
    number_of_images_to_classify = 2
    kmeans_knn_statistics(train_imgs, train_class_labels, cropped_images, number_of_images_to_classify,
                      color_labels, class_labels, 5, opt, False, False, True)

    """
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
    """


def visualize_statistics_fisher(statistics):
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    num_images = len(statistics)
    colors = plt.cm.get_cmap('tab10', num_images)

    # Inicializar listas para guardar las medias
    Ks = [stat['K'] for stat in statistics[0]]
    fisher_avg = []

    # Calcular la media para cada K
    for i in range(len(Ks)):
        fisher_avg.append(np.mean([image_stats[i]['FISHER'] for image_stats in statistics]))

    for idx, image_stats in enumerate(statistics):
        # Extraer los valores de las estadísticas para cada K
        fisher = [stat['FISHER'] for stat in image_stats]

        # Graficar WCD vs K
        axs[0].plot(Ks, fisher, marker='o', label=f'Image {idx + 1}', color=colors(idx))
        axs[0].set_title('Fisher Coeficient vs K', fontsize=10)
        axs[0].set_xlabel('Number of Clusters (K)', fontsize=8)
        axs[0].set_ylabel('Fisher Coeficient', fontsize=8)
        axs[0].tick_params(axis='both', which='major', labelsize=8)
        axs[0].grid(True)

    # Gráficos promedio
    axs[1].plot(Ks, fisher_avg, marker='o', label='Average', color='black')
    axs[1].set_title('Fisher Coeficient vs K (Average)', fontsize=10)
    axs[1].set_xlabel('Number of Clusters (K)', fontsize=8)
    axs[1].set_ylabel('Fisher Coeficient', fontsize=8)
    axs[1].tick_params(axis='both', which='major', labelsize=8)
    axs[1].grid(True)

    # Añadir leyenda a cada gráfico
    for ax in axs:
        ax.legend()

    # Ajustar espacio entre gráficos
    plt.tight_layout(pad=4.0)

    # Mostrar los gráficos
    plt.show()


def visualize_statistics(statistics):
    fig, axs = plt.subplots(2, 3, figsize=(14, 7))

    num_images = len(statistics)
    colors = plt.cm.get_cmap('tab10', num_images)

    # Inicializar listas para guardar las medias
    Ks = [stat['K'] for stat in statistics[0]]
    WCDs_avg = []
    convergence_times_avg = []
    color_accuracy_avg = []

    # Calcular la media para cada K
    for i in range(len(Ks)):
        WCDs_avg.append(np.mean([image_stats[i]['WCD'] for image_stats in statistics]))
        convergence_times_avg.append(np.mean([image_stats[i]['Convergence_time'] for image_stats in statistics]))
        color_accuracy_avg.append(np.mean([image_stats[i]['Color_accuracy'] for image_stats in statistics]))

    for idx, image_stats in enumerate(statistics):
        # Extraer los valores de las estadísticas para cada K
        WCDs = [stat['WCD'] for stat in image_stats]
        convergence_times = [stat['Convergence_time'] for stat in image_stats]
        color_accuracy = [stat['Color_accuracy'] for stat in image_stats]

        # Graficar WCD vs K
        axs[0, 0].plot(Ks, WCDs, marker='o', label=f'Image {idx + 1}', color=colors(idx))
        axs[0, 0].set_title('Within-Class-Distance (WCD) vs K', fontsize=10)
        axs[0, 0].set_xlabel('Number of Clusters (K)', fontsize=8)
        axs[0, 0].set_ylabel('WCD', fontsize=8)
        axs[0, 0].tick_params(axis='both', which='major', labelsize=8)
        axs[0, 0].grid(True)
        # Graficar tiempo de convergencia vs K
        axs[0, 1].plot(Ks, convergence_times, marker='o', label=f'Image {idx + 1}', color=colors(idx))
        axs[0, 1].set_title('Convergence Time vs K', fontsize=10)
        axs[0, 1].set_xlabel('Number of Clusters (K)', fontsize=8)
        axs[0, 1].set_ylabel('Convergence Time (seconds)', fontsize=8)
        axs[0, 1].tick_params(axis='both', which='major', labelsize=8)
        axs[0, 1].grid(True)
        # Graficar precisión vs K
        axs[0, 2].plot(Ks, color_accuracy, marker='o', label=f'Image {idx + 1}', color=colors(idx))
        axs[0, 2].set_title('Accuracy vs K', fontsize=10)
        axs[0, 2].set_xlabel('Number of Clusters (K)', fontsize=10)
        axs[0, 2].set_ylabel('Accuracy (%)', fontsize=10)
        axs[0, 2].tick_params(axis='both', which='major', labelsize=8)
        axs[0, 2].grid(True)

    # Gráficos promedio
    axs[1, 0].plot(Ks, WCDs_avg, marker='o', label='Average', color='black')
    axs[1, 0].set_title('Within-Class-Distance (WCD) vs K (Average)', fontsize=10)
    axs[1, 0].set_xlabel('Number of Clusters (K)', fontsize=8)
    axs[1, 0].set_ylabel('WCD', fontsize=8)
    axs[1, 0].tick_params(axis='both', which='major', labelsize=8)
    axs[1, 0].grid(True)

    axs[1, 1].plot(Ks, convergence_times_avg, marker='o', label='Average', color='black')
    axs[1, 1].set_title('Convergence Time vs K (Average)', fontsize=10)
    axs[1, 1].set_xlabel('Number of Clusters (K)', fontsize=8)
    axs[1, 1].set_ylabel('Convergence Time (seconds)', fontsize=8)
    axs[1, 1].tick_params(axis='both', which='major', labelsize=8)
    axs[1, 1].grid(True)

    axs[1, 2].plot(Ks, color_accuracy_avg, marker='o', label='Average', color='black')
    axs[1, 2].set_title('Accuracy vs K (Average)', fontsize=10)
    axs[1, 2].set_xlabel('Number of Clusters (K)', fontsize=10)
    axs[1, 2].set_ylabel('Accuracy (%)', fontsize=10)
    axs[1, 2].tick_params(axis='both', which='major', labelsize=8)
    axs[1, 2].grid(True)

    # Añadir leyenda a cada gráfico
    for ax in axs.flat:
        ax.legend()

    # Ajustar espacio entre gráficos
    plt.tight_layout(pad=4.0)

    # Mostrar los gráficos
    plt.show()


def print_statistics(statistic):
    for key, value in statistic.items():
        print(f'{key}: {value}')
    print()
    
def test_best_K(crop=False, first=None, last=None, t=None, maxK=10):
    print("_________________________Best_K___________________________")
    # TESTS BASELINE SIN TOCAR
    """
    opt1 = {
    'km_init': 'kmeans++'
    }
    opt2 = {
        'km_init': 'random'
    }
    
    # Base img_01 -->
    print("____________________IMAGE 01_____________________")
    print("Base:")
    images_to_classify = cropped_images[0:1]
    kmeans_statistics_nonRandom(train_imgs, train_class_labels, images_to_classify,
                      color_labels, class_labels, 10, True, True, True, options=None)
    # img_01 con kmeans++
    print("----------------w/kmeans++:----------------")
    images_to_classify = cropped_images[0:1]
    kmeans_statistics_nonRandom(train_imgs, train_class_labels, images_to_classify,
                      color_labels, class_labels, 10, True, True, True, options=opt1)
    # img_01 con random
    print("----------------w/random:----------------")
    images_to_classify = cropped_images[0:1]
    kmeans_statistics_nonRandom(train_imgs, train_class_labels, images_to_classify,
                      color_labels, class_labels, 10, True, True, True, options=opt2)
  
    # Base img_02 -->
    print("____________________IMAGE 02_____________________")
    print("----------------Base:----------------")
    images_to_classify = cropped_images[1:2]
    kmeans_statistics_nonRandom(train_imgs, train_class_labels, images_to_classify,
                      color_labels, class_labels, 10, True, True, True, options=None)
    # img_02 con kmeans++
    print("----------------w/kmeans++:----------------")
    images_to_classify = cropped_images[1:2]
    kmeans_statistics_nonRandom(train_imgs, train_class_labels, images_to_classify,
                      color_labels, class_labels, 10, True, True, True, options=opt1)
    # img_02 con random
    print("----------------w/random:----------------")
    images_to_classify = cropped_images[1:2]
    kmeans_statistics_nonRandom(train_imgs, train_class_labels, images_to_classify,
                      color_labels, class_labels, 10, True, True, True, options=opt2)
 
    # Base img_03 -->
    print("____________________IMAGE 03_____________________")
    print("----------------Base:----------------")
    images_to_classify = cropped_images[2:3]
    kmeans_statistics_nonRandom(train_imgs, train_class_labels, images_to_classify,
                      color_labels, class_labels, 10, True, True, True, options=None)
    # img_03 con kmeans++
    print("----------------w/kmeans++:----------------")
    images_to_classify = cropped_images[2:3]
    kmeans_statistics_nonRandom(train_imgs, train_class_labels, images_to_classify,
                      color_labels, class_labels, 10, True, True, True, options=opt1)
    # img_03 con random
    print("----------------w/random:----------------")
    images_to_classify = cropped_images[2:3]
    kmeans_statistics_nonRandom(train_imgs, train_class_labels, images_to_classify,
                      color_labels, class_labels, 10, True, True, True, options=opt2)

    # Base img_01 -->
    print("____________________IMAGE 01_____________________")
    print("Base:")
    images_to_classify = test_imgs[0:1]
    kmeans_statistics_nonRandom(train_imgs, train_class_labels, images_to_classify,
                      color_labels, class_labels, 10, True, True, True, options=None)
    # img_01 con kmeans++
    print("----------------w/kmeans++:----------------")
    images_to_classify = test_imgs[0:1]
    kmeans_statistics_nonRandom(train_imgs, train_class_labels, images_to_classify,
                      color_labels, class_labels, 10, True, True, True, options=opt1)
    # img_01 con random
    print("----------------w/random:----------------")
    images_to_classify = test_imgs[0:1]
    kmeans_statistics_nonRandom(train_imgs, train_class_labels, images_to_classify,
                      color_labels, class_labels, 10, True, True, True, options=opt2)
  
    # Base img_02 -->
    print("____________________IMAGE 02_____________________")
    print("Base:")
    images_to_classify = test_imgs[1:2]
    kmeans_statistics_nonRandom(train_imgs, train_class_labels, images_to_classify,
                      color_labels, class_labels, 10, True, True, True, options=None)
    # img_02 con kmeans++
    print("----------------w/kmeans++:----------------")
    images_to_classify = test_imgs[1:2]
    kmeans_statistics_nonRandom(train_imgs, train_class_labels, images_to_classify,
                      color_labels, class_labels, 10, True, True, True, options=opt1)
    # img_02 con random
    print("----------------w/random:----------------")
    images_to_classify = test_imgs[1:2]
    kmeans_statistics_nonRandom(train_imgs, train_class_labels, images_to_classify,
                      color_labels, class_labels, 10, True, True, True, options=opt2)


    # Base img_03 -->
    print("____________________IMAGE 03_____________________")
    print("----------------Base:----------------")
    images_to_classify = test_imgs[2:3]
    kmeans_statistics_nonRandom(train_imgs, train_class_labels, images_to_classify,
                      color_labels, class_labels, 10, True, True, True, options=None)
    # img_03 con kmeans++
    print("----------------w/kmeans++:----------------")
    images_to_classify = test_imgs[2:3]
    kmeans_statistics_nonRandom(train_imgs, train_class_labels, images_to_classify,
                      color_labels, class_labels, 10, True, True, True, options=opt1)
    # img_03 con random
    print("----------------w/random:----------------")
    images_to_classify = test_imgs[2:3]
    kmeans_statistics_nonRandom(train_imgs, train_class_labels, images_to_classify,
                      color_labels, class_labels, 10, True, True, True, options=opt2)
    
    """
    
    if(t == 15):
        t = { 'threshold': 15 }
    elif (t == 10):
        t = { 'threshold': 10 }
    elif (t == 25):
        t = { 'threshold': 25 }
    elif (t == 30):
        t = { 'threshold': 30 }
    else:
        t = { 'threshold': 20 }

    if (crop == True):
        crop = "cropped image"
        print("Using: ", crop)
        print("Threashold", t)
        for x in range(first,last):
            print("-----------Image nº",x)
            images_to_classify = cropped_images[x:x+1]
            kmeans_statistics_nonRandom_Specific(train_imgs, train_class_labels, images_to_classify,
                              color_labels, class_labels, maxK, True, True, True, options=t)
    else:
        crop = "un-cropped image"
        print("Using: ", crop)
        print("Threashold", t)
        for x in range(first,last):
            print("-----------Image nº",x)
            images_to_classify = test_imgs[x:x+1]
            kmeans_statistics_nonRandom_Specific(train_imgs, train_class_labels, images_to_classify,
                              color_labels, class_labels, maxK, True, True, True, options=t)

    

def MyPlot3DCloud(km, rows=1, cols=1, spl_id=1, title=''):
    ax = plt.gcf().add_subplot(rows, cols, spl_id, projection='3d')

    for k in range(km.K):
        Xl = km.X[km.labels == k, :]
        ax.scatter(
            Xl[:, 0], Xl[:, 1], Xl[:, 2], marker='.', c=km.centroids[np.ones((Xl.shape[0]), dtype='int') * k, :] / 255
        )

    plt.xlabel('dim 1')
    plt.ylabel('dim 2')
    ax.set_zlabel('dim 3')
    ax.set_title(title)
    return ax

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
    # test_retrieval_combined(train_imgs[:300], train_color_labels[:300], train_class_labels[:300])

    """Tests kmeans_statistics"""
    #test_kmeans_statistics()
    test_best_K(crop=True, first=0, last=1, t=10, maxK=10)
    
    

    print("____________________IMAGE 01_____________________")
    print("Base:")
    images_to_classify = cropped_images[:5]
    kmeans_statistics_nonRandom_plusF(train_imgs, train_class_labels, images_to_classify,
                       color_labels, class_labels, 5, True, False, True, options=None)

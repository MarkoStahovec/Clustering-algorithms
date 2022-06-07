import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from utils import sqrt
from utils import copy
from utils import k_means
from utils import agglomerative
from utils import divisive
from utils import generate_initial_points
from utils import generate_all_points
from utils import split_into_subsets
from utils import print_cluster_stats
from utils import clean_third_indices


labels = ["K-Centroid", "K-Medoid", "Agglomerative Clustering", "Divisive Clustering"]
sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1,
        color_codes=False, rc=None)
sns.set_palette(sns.color_palette(
    ["#2C4BC6", "#D73F3F", "#1AA51E", "#7D20B3", "#E6AF2D",
     "#CCD736", "#B72199", "#1D9EB7", "#1DB674", "#7F7F7F",
     "#7A99FD", "#FF8989", "#83FF83", "#FFDF86", "#D492FF",
     "#89EAEE", "#FF9CF3", "#FCFF94", "#8EBDA6", "#1D5E9F"]))


# cleans given subset of index 2, so there are coordinates only
def clean_subset(subset):
    current = [x[:] for x in subset]
    for row in current:
        del row[2]

    current = np.array(current)
    return current


# plots data after single algorithm run
def plot_data(data, k_clusters, i_centres, f_centres):
    subsets = split_into_subsets(data, k_clusters)
    for i in range(len(subsets)):
        current_set = clean_subset(subsets[i])
        sns.scatterplot(x=current_set[:, 0], y=current_set[:, 1])

    """
    i_centres = np.array(i_centres)
    sns.scatterplot(x=i_centres[:, 0], y=i_centres[:, 1], marker="2", color="aqua",
                    linewidth=3, s=100, hue_norm=(0, len(i_centres)), label="Initial Centres")
    """

    clean_third_indices(f_centres)
    f_centres = np.array(f_centres)
    sns.scatterplot(x=f_centres[:, 0], y=f_centres[:, 1], marker="x", color="black",
                    linewidth=3, s=70, hue_norm=(0, len(f_centres)), label="Final Centres")

    plt.show()


# plots data when multiple algorithms have run
def plot_all_algorithms(data, k_clusters, i_centres, f_centres):
    fig, axis = plt.subplots(2, 2, figsize=(14, 8))

    x = 0
    for j in range(len(axis)):
        for i in range(len(axis[j])):

            subsets = split_into_subsets(data[x], k_clusters)
            for y in range(len(subsets)):
                current_set = clean_subset(subsets[y])
                sns.scatterplot(x=current_set[:, 0], y=current_set[:, 1], ax=axis[j][i])

            if j == 0:
                ini_centres = np.array(i_centres[x])
                sns.scatterplot(x=ini_centres[:, 0], y=ini_centres[:, 1], marker="2", color="b", linewidth=3,
                                s=100, label="Initial Centres", ax=axis[j][i])

            clean_third_indices(f_centres[x])
            fin_centres = np.array(f_centres[x])
            sns.scatterplot(x=fin_centres[:, 0], y=fin_centres[:, 1], marker="x", color="black", linewidth=3,
                            s=70, label="Final Centres", ax=axis[j][i]).set_title(labels[x])

            x += 1
    plt.show()


# adds results of an algorithm into database of all results for later display
def add_into_base(init_base, final_base, all_base, init_a, final_a, all_a):
    init_base.append(copy.deepcopy(init_a))
    final_base.append(copy.deepcopy(final_a))
    all_base.append(copy.deepcopy(all_a))

    return init_base, final_base, all_base


# handles input so it doesnt contain any bogus values
def input_handler():
    while True:
        start_p_ = input("[INT] [INPUT] -- Number of generating points: ")
        if start_p_.isnumeric() and int(start_p_) >= 1:
            start_p_ = int(start_p_)
            break
        else:
            print("[ERROR] -- Incorrect input.")

    while True:
        all_p_ = input("[INT] [INPUT] -- Number of all points: ")
        if all_p_.isnumeric() and int(all_p_) >= start_p_:
            all_p_ = int(all_p_)
            break
        else:
            print("[ERROR] -- Incorrect input.")

    while True:
        offset_ = input("[INT] [INPUT] -- Offset for generated points: ")
        if offset_.isnumeric() and int(offset_) >= 1:
            offset_ = int(offset_)
            break
        else:
            print("[ERROR] -- Incorrect input.")

    while True:
        dimension_ = input("[INT] [INPUT] -- Dimension of space: ")
        if dimension_.isnumeric() and int(dimension_) > sqrt(all_p_):
            dimension_ = int(dimension_)
            break
        else:
            print("[ERROR] -- Incorrect input.")

    while True:
        clusters_ = input("[INT] [INPUT] -- Number of clusters: ")
        if clusters_.isnumeric() and int(clusters_) >= 1:
            clusters_ = int(clusters_)
            break
        else:
            print("[ERROR] -- Incorrect input.")

    return start_p_, all_p_, dimension_, clusters_, offset_


if __name__ == '__main__':
    start_p, all_p, dimension, clusters, offset = input_handler()

    init_centres_a = []
    final_centres_a = []
    all_points_a = []

    init_points = generate_initial_points(dimension, start_p, offset)
    all_points_base = generate_all_points(dimension, init_points, all_p, offset)

    all_points = copy.deepcopy(all_points_base)
    init_centres, final_centres, all_points = k_means(all_points, clusters, "centroid")
    print_cluster_stats(all_points, final_centres, clusters, labels[0])
    init_centres_a, final_centres_a, all_points_a = add_into_base(init_centres_a, final_centres_a, all_points_a,
                                                                  init_centres, final_centres, all_points)
    all_points = copy.deepcopy(all_points_base)
    init_centres, final_centres, all_points = k_means(all_points, clusters, "medoid")
    print_cluster_stats(all_points, final_centres, clusters, labels[1])
    init_centres_a, final_centres_a, all_points_a = add_into_base(init_centres_a, final_centres_a, all_points_a,
                                                                  init_centres, final_centres, all_points)

    all_points = copy.deepcopy(all_points_base)
    init_centres, final_centres, all_points = agglomerative(all_points, clusters)
    print_cluster_stats(all_points, final_centres, clusters, labels[2])
    init_centres_a, final_centres_a, all_points_a = add_into_base(init_centres_a, final_centres_a, all_points_a,
                                                                  init_centres, final_centres, all_points)

    all_points = copy.deepcopy(all_points_base)
    init_centres, final_centres, all_points = divisive(all_points, clusters)
    print_cluster_stats(all_points, final_centres, clusters, labels[3])
    init_centres_a, final_centres_a, all_points_a = add_into_base(init_centres_a, final_centres_a, all_points_a,
                                                                  init_centres, final_centres, all_points)

    # plot_data(all_points, clusters, init_centres, final_centres)
    plot_all_algorithms(all_points_a, clusters, init_centres_a, final_centres_a)

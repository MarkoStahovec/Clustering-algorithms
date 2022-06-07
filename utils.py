import copy
from math import sqrt
from random import randrange


# computes euclidian distance of two given points
def euclidian_distance(x, y):
    return sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))


# cleans given subset of index 2, so there are coordinates only
def clean_third_indices(f_cent):
    for row in f_cent:
        if len(row) == 3:
            row.pop(-1)


# splits given data into multiple subsets according to their respective cluster
def split_into_subsets(data, N):
    split_sets = []  # all sets
    x = 0  # cluster index
    mainset = sorted(data, key=lambda l: l[2])  # sort all points according to their clusters

    for i in range(N):
        temp_list = []
        if x >= len(mainset):  # if all points are done
            break

        set_index = mainset[x][2]
        while mainset[x][2] == set_index:  # cycle through all points within same clusters
            temp_list.append(mainset[x])  # append current point
            x += 1
            if x >= len(mainset):
                break

        if temp_list:  # append all points within same cluster to list of all points
            split_sets.append(temp_list)

    return split_sets


# generates all initial points, from which all other points will be generated
def generate_initial_points(dimension, N, offset):
    start_points = []
    for i in range(N):
        start_points.append([randrange(- dimension + offset, dimension - offset),
                             randrange(- dimension + offset, dimension - offset)])

    return start_points


# generates all points given list of starting points
def generate_all_points(dimension, start_points, N, offset):
    all_points = []
    i = spreader = 0
    rand_start_point = start_points[randrange(0, len(start_points))]  # append first random point with given offset
    new_point = [randrange(rand_start_point[0] - offset, rand_start_point[0] + offset),
                 randrange(rand_start_point[1] - offset, rand_start_point[1] + offset)]
    all_points.append(new_point)

    while i < N - 1:
        if randrange(0, len(all_points)) > randrange(0, N + spreader):  # statement to randomly pick from either list
            rand_start_point = start_points[randrange(0, len(start_points))]
        else:
            rand_start_point = all_points[randrange(0, len(all_points))]
            spreader += 1  # helping variable to spread points more randomly

        new_point = [randrange(rand_start_point[0] - offset, rand_start_point[0] + offset),
                     randrange(rand_start_point[1] - offset, rand_start_point[1] + offset)]

        if new_point[0] > dimension or new_point[0] < -dimension \
                or new_point[1] > dimension or new_point[1] < -dimension:  # check for dimensions
            continue

        if new_point not in all_points:  # if its unique, add it
            all_points.append(new_point)
            i += 1

    return all_points


# picks few random initial centres for kmeans
def create_initial_centres(all_points, k_clusters):
    init_centroids = []
    while len(init_centroids) != k_clusters:
        rand_point = all_points[randrange(0, len(all_points))]
        if rand_point not in init_centroids:
            init_centroids.append(rand_point)
    return init_centroids


def print_cluster_stats(points, centres, k_clusters, label):
    # clean_third_indices(centres)
    cluster_distances = []

    subsets = split_into_subsets(points, k_clusters)
    for i in range(len(subsets)):
        curr_eval = 0
        for j in range(len(subsets[i])):
            curr_eval += euclidian_distance(centres[i], subsets[i][j])

        curr_eval = curr_eval / len(subsets[i])
        cluster_distances.append(round(curr_eval, 3))

    print(f"\n---------- {label} ----------")
    for i in range(len(cluster_distances)):
        print(f"Cluster {i} avg. distance: {cluster_distances[i]}")


# recalculates new values for medoids for all clusters
def calculate_medoids(points, clusters):
    new_medoids = []

    subsets = split_into_subsets(points, clusters)
    for i in range(len(subsets)):  # cycle through all clusters
        best_eval = 0xFFFFFFFF  # initial maximum evaluation
        best_medoid = None
        curr_subset = subsets[i]
        for j in range(len(curr_subset)):  # for all points compute its distance to all other points
            curr_eval = 0
            for k in range(len(curr_subset)):
                if j != k:
                    curr_eval += euclidian_distance(curr_subset[j], curr_subset[k])

            if curr_eval < best_eval:  # if better evaluation for medoid is found
                best_eval = curr_eval
                best_medoid = curr_subset[j]

        new_medoids.append(best_medoid)

    return new_medoids


# recalculates new values for centroids for all clusters
def calculate_centroids(points, clusters):
    new_centroids = []

    subsets = split_into_subsets(points, clusters)
    for i in range(len(subsets)):  # cycle through clusters
        sum_x = sum_y = 0
        for j in range(len(subsets[i])):  # cycle through points within clusters
            sum_x += subsets[i][j][0]
            sum_y += subsets[i][j][1]
        new_centroids.append([round(sum_x / len(subsets[i]), 2), round(sum_y / len(subsets[i]), 2)])

    return new_centroids


# assigns all points to their respective clusters for newly computed coordinates
def assign_to_clusters(centroids, points):
    # points = [[5, 4], [20, 24]]
    # start_points = [[10, 12], [18, 20]]

    for point in points:  # cycle through all points
        shortest_distance = 0xFFFFFFFF
        shortest_index = 0

        for i in range(len(centroids)):
            distance = euclidian_distance(point, centroids[i])
            if distance < shortest_distance:  # if shorter distance is found
                shortest_distance = distance
                shortest_index = i

        if len(point) > 2:  # if cluster was assigned before, erase it
            point.pop(2)
        point.append(shortest_index)

    return points


# finds smallest distance between two points in distance matrix
def get_smallest_distance(matrix):
    x_idx = y_idx = 0
    smallest_dist = 0xFFFFFFFF

    for i in range(len(matrix)):  # cycle through rows
        if not matrix[i]:  # if zero is present, skip it, so self distance is not taken into account
            continue
        for j in range(i+1, len(matrix)):  # cycle through all cols
            if matrix[i][j] < smallest_dist and i != j:
                smallest_dist = matrix[i][j]
                x_idx = i
                y_idx = j

    return x_idx, y_idx


# calculates new centroid from list of given points
def get_new_centroids(points):
    new_centroids = []

    for i in range(len(points)):
        sum_x = sum_y = 0
        for j in range(len(points)):
            sum_x += points[j][0]
            sum_y += points[j][1]
        new_centroids.append([round(sum_x / len(points), 2), round(sum_y / len(points), 2)])

    return new_centroids


# checks whether all clusters are equal and returns boolean accordingly
def check_equality_in_clusters(current, previous):
    for i in range(len(current)):
        if current[i][2] != previous[i][2]:
            return False
    return True


# placeholder function for kmeans algorithm
def k_means(all_points, k_clusters, recalc_method):
    init_centres = create_initial_centres(all_points, k_clusters)
    current_eval = assign_to_clusters(init_centres, all_points)

    while True:
        if recalc_method == "centroid":
            new_centres = calculate_centroids(all_points, k_clusters)
        else:
            new_centres = calculate_medoids(all_points, k_clusters)

        previous_eval = copy.deepcopy(current_eval)  # copy current evaluation into another variable so its comparable
        current_eval = assign_to_clusters(new_centres, all_points)

        if check_equality_in_clusters(current_eval, previous_eval):
            break

    return init_centres, new_centres, all_points


# assigns all remaining points to their respective clusters
def assign_remaining_points(source, destination, all_p):
    for i in range(len(source)):
        destination[i].append(source[i][0])

    for i in range(len(destination)):
        for j in range(len(destination[i])):
            idx = all_p.index(destination[i][j])
            all_p[idx].append(i)


"""
def agglomerative(all_points, k_clusters):
    # all_points = [[-2, 0], [-3, 1], [-1, -1], [-2, 1], [-3, 0], [-3, 2], [-2, -1], [-4, -1], [-1, 1], [1, 1]]
    init_centres = []
    # init_centres = create_initial_centres(all_points, k_clusters)

    remaining_points = []
    all_clusters = []
    for i in range(len(all_points)):
        remaining_points.append([all_points[i]])
        all_clusters.append(all_points[i])

    # final_clusters = [[] for _ in range(k_clusters)]
    distance_matrix = [[0.0 for _ in range(len(all_points))] for _ in range(len(all_points))]
    for i in range(len(all_points)):
        for j in range(len(all_points)):
            if i != j:
                distance_matrix[i][j] = round(euclidian_distance(all_points[i], all_points[j]), 2)

    while len(all_clusters) != k_clusters:
        x_index, y_index = get_smallest_distance(distance_matrix)
        all_clusters[x_index] = get_new_centroids([remaining_points[x_index][0],
                                                   remaining_points[y_index][0]])[0]

        for j in range(len(distance_matrix[x_index])):  # updates values in current row with new centroid
            distance_matrix[x_index][j] = round(euclidian_distance(all_clusters[x_index], all_clusters[j]), 2)

        for i in range(0, len(distance_matrix)):  # updates desired column values and removes column that is merged
            distance_matrix[i][x_index] = round(euclidian_distance(all_clusters[x_index], all_clusters[i]), 2)
            del distance_matrix[i][y_index]

        # final_clusters[x_index].append(remaining_points[y_index][0])
        remaining_points[x_index][0] = all_clusters[x_index]
        del all_clusters[y_index]
        del distance_matrix[y_index]
        del remaining_points[y_index]

    # assign_remaining_points(remaining_points, final_clusters, all_points)

    all_points = assign_to_clusters(all_clusters, all_points)
    all_clusters = calculate_centroids(all_points, k_clusters)
    all_points = assign_to_clusters(all_clusters, all_points)
    return init_centres, all_clusters, all_points
"""


# assigns all points to individual clusters
def assign_all_to_individual_clusters(all_p):
    for i in range(len(all_p)):
        all_p[i].append(i)


# placeholder function for agglomerative clustering algorithm
def agglomerative(all_points, k_clusters):
    # all_points = [[-2, 0], [-3, 1], [-1, -1], [-2, 1], [-3, 0], [-3, 2], [-2, -1], [-4, -1], [-1, 1], [1, 1]]
    assign_all_to_individual_clusters(all_points)

    init_centres = []
    remaining_points = []  # all points stored according to their clusters
    all_clusters = []  # cluster values for centers
    for i in range(len(all_points)):
        remaining_points.append([all_points[i]])
        all_clusters.append(all_points[i])

    # final_clusters = [[] for _ in range(k_clusters)]
    distance_matrix = [[0.0 for _ in range(len(all_points))] for _ in range(len(all_points))]
    for i in range(len(all_points)):  # create distance matrix
        for j in range(len(all_points)):
            if i != j:
                distance_matrix[i][j] = round(euclidian_distance(all_points[i], all_points[j]), 2)

    while len(all_clusters) != k_clusters:  # do while number of clusters is not equal to desired number of clusters
        # print(len(all_clusters))
        x_index, y_index = get_smallest_distance(distance_matrix)  # find current smallest distance
        all_clusters[x_index] = get_new_centroids(remaining_points[x_index] + remaining_points[y_index])[0]

        for j in range(len(distance_matrix[x_index])):  # updates values in current row with new centroid
            distance_matrix[x_index][j] = round(euclidian_distance(all_clusters[x_index], all_clusters[j]), 2)

        for i in range(0, len(distance_matrix)):  # updates desired column values and removes column that is merged
            distance_matrix[i][x_index] = round(euclidian_distance(all_clusters[x_index], all_clusters[i]), 2)
            del distance_matrix[i][y_index]

        # final_clusters[x_index].append(remaining_points[y_index][0])
        # remaining_points[x_index][0] = all_clusters[x_index]
        remaining_points[x_index].extend(remaining_points[y_index])  # merge points into one cluster
        del all_clusters[y_index]  # delete remains
        del distance_matrix[y_index]
        del remaining_points[y_index]

    # assign_remaining_points(remaining_points, final_clusters, all_points)

    all_points = assign_to_clusters(all_clusters, all_points)
    return init_centres, all_clusters, all_points


# calculate new centres for divisive algorithm
def calculate_divisive_centroids(points, clusters, init_centres):
    new_centroids = calculate_centroids(points, clusters)

    subsets = split_into_subsets(points, clusters)
    for i in range(clusters):  # extra cycle to append empty list for indices without points yet
        if subsets[i][0][2] == i:
            continue
        else:
            subsets.insert(i, [[]])  # append empty 2d list
            new_centroids.insert(i, init_centres[i])

    return new_centroids


# kmeans version for divisive algorithm to evaluate points belonging to clusters
def divisive_kmeans(all_points, k_clusters, init_centres):
    current_eval = assign_to_clusters(init_centres, all_points)

    while True:  # do while there is no change in clusters
        new_centres = calculate_divisive_centroids(all_points, k_clusters, init_centres)
        previous_eval = copy.deepcopy(current_eval)
        current_eval = assign_to_clusters(new_centres, all_points)

        if check_equality_in_clusters(current_eval, previous_eval):
            break

    return new_centres, all_points


# retrieve the longest distance between all points within cluster
def get_longest_distance(all_points):
    index = 0
    longest = -1

    for j in range(len(all_points)):
        curr_distance = 0
        for k in range(len(all_points)):
            if all_points[j][2] != all_points[k][2]:
                continue
            curr_distance += euclidian_distance(all_points[j], all_points[k])
        if curr_distance > longest:
            longest = curr_distance
            index = j

    return index


# placeholder function for divisive clustering algorithm
def divisive(all_points, k_clusters):
    # all_points = [[-1, -3], [-1, -5], [-2, -2], [1, -5], [2, -2]]
    all_centers = [all_points[randrange(0, len(all_points))].copy()]
    # all_centers = [all_points[2].copy()]
    init_center = all_centers.copy()
    all_points = assign_to_clusters(all_centers, all_points)  # assign all points to a single cluster
    all_centers = calculate_centroids(all_points, 1)

    while len(all_centers) != k_clusters:  # while number of clusters is not desired amount
        longest_distance_index = get_longest_distance(all_points)  # get point that is the most dissimilar
        all_centers.append(all_points[longest_distance_index].copy())  # copy it into the list of centers

        subsets = split_into_subsets(all_points, len(all_centers))
        new_clusters, new_points = divisive_kmeans(subsets[all_points[longest_distance_index][2]],
                                                   len(all_centers), all_centers)
        all_centers = new_clusters.copy()  # overwrite previous centers with new values

    return init_center, all_centers, all_points

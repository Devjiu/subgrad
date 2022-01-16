import numpy as np


def generate_points_to_cover(center, radius, number_of_points):
    random_points = np.random.randn(len(center), number_of_points-1)
    correct_points = []
    # # print("raw rand: ", random_points)
    # point = random_points[:, 0]
    # # print("p0: ", point)
    # point = (point / np.linalg.norm(point)) * radius
    # # to have at least single point with max radius
    # correct_points.append(point + center)
    for ind in range(0, number_of_points-1):
        point = random_points[:, ind]
        point = (point / np.linalg.norm(point)) * radius
        correct_points.append(point + center)
    print("norm: ", np.linalg.norm(center - correct_points[-1]))
    # print("center: ", center)
    # print("corr l: ", correct_points[-1])
    # print("diam: ", 2 * center - correct_points[-1])
    correct_points.append(2 * center - correct_points[-1])
    # print("Generated shape: ", np.array(correct_points).shape)
    # print("First point: ", correct_points[0])
    return np.array(correct_points)

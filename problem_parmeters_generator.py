import numpy as np


def generate_points_to_cover(center, radius, number_of_points):
    random_points = np.random.randn(len(center), number_of_points)
    correct_points = []
    # print("raw rand: ", random_points)
    point = random_points[:, 0]
    # print("p0: ", point)
    point = (point / np.linalg.norm(point)) * radius
    # to have at least single point with max radius
    correct_points.append(point + center)
    for ind in range(1, number_of_points):
        point = random_points[:, ind]
        if np.linalg.norm(point - center) > radius:
            point = (point / np.linalg.norm(point)) * np.random.randint(0, radius)
        correct_points.append(point + center)
    # print("Generated shape: ", np.array(correct_points).shape)
    # print("First point: ", correct_points[0])
    return correct_points

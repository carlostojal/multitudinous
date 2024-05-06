# Point cloud processing utilities

import numpy as np

def farthest_point_sampling(pointcloud: np.array[np.array], n_points: int = 1024):

    """
    Farthest point sampling algorithm to downsample a point cloud
    """

    # verify if the number of points is already larger than the threshold
    if n_points >= pointcloud.shape[0]:
        return pointcloud
    
    # initialize the list of selected points
    selected_points = []

    # select the first point randomly
    selected_points.append(np.random.randint(pointcloud.shape[0]))

    # calculate the distance from the selected point to all the other points
    # the norm of the difference is the Euclidean distance
    distances = np.linalg.norm(pointcloud - pointcloud[selected_points[0]], axis=1)
    
    # iterate over the number of points to select
    for i in range(n_points - 1):
        
        # select the point with the maximum distance to the selected points
        # the next selected point is the one with the maximum distance
        selected_points.append(np.argmax(distances))
        
        # calculate the distance from the selected point to all the other points
        # the new distances are the element-wise minimum between the current distances and the distance to the new point
        distances = np.minimum(distances, np.linalg.norm(pointcloud - pointcloud[selected_points[-1]], axis=1))

    return pointcloud[selected_points]

import numpy as np
from numpy import linalg as la
from numpy.random import normal
import matplotlib.pyplot as plt
import time

def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print ('%r %2.2f sec' % (method.__name__, te-ts))
        return result

    return timed

def vp_tree(data,metric):
    '''
    data (numpy array): input points
    metric(function): distance between two points

    TODO: obtain improvement by using two distances,
    'distance_left' =  metric(close_points[-1],D['point'])
    'distance_right' =  metric(far_points[0],D['point'])
    '''
    if len(data) == 0:
        return None
    elif len(data) == 1:
        D = {}
        D['point'] = data[0]
        D['distance'] = None
        D['children'] = None
        return D
    else:
        D = {}
        D['point'] = data[0] ## randomize?
        sorted_points_from_point = sorted(data[1:],key = lambda p: metric(D['point'],p))
        close_points,far_points = np.array_split(sorted_points_from_point,2)
        D['distance'] = metric(close_points[-1],D['point'])

        D['children'] = []
        for points in close_points,far_points:
            new_tree = vp_tree(points, metric)
            D['children'].append(new_tree)
        return D

@timeit
def vp_tree_wrapper(data,metric):
    return vp_tree(data,metric)

def find_within_epsilon(vp_tree,metric,point,epsilon,found_points = None):
    if found_points is None:
        found_points = []

    root = vp_tree['point']
    if root is None:
        return found_points

    distance_root_to_point = metric(root,point)
    if distance_root_to_point <= epsilon:
        found_points.append(root)

    cutoff_distance = vp_tree['distance']
    if cutoff_distance is None:
        return found_points

    if distance_root_to_point - cutoff_distance <= epsilon:
        left_child = vp_tree['children'][0]
        if not left_child is None:
            found_points += find_within_epsilon(left_child,metric,point,epsilon)

    if cutoff_distance - distance_root_to_point <= epsilon:
        right_child = vp_tree['children'][1]
        if not right_child is None:
            found_points += find_within_epsilon(right_child,metric,point,epsilon)

    return found_points

if __name__ == "__main__":
    metric = lambda x,y: la.norm(x-y)
    tree = vp_tree_wrapper(np.array([1,2,3,4,5,6]),metric)
    print(tree)
    print(find_within_epsilon(tree,metric,2.,2.9999))

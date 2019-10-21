#!/usr/bin/env python

# utilities file - includes the common functions
import io
import numpy as np
from numpy.linalg import norm

# Converts planar polar coordinates (radius and angle) to cartesian coordinate
def polar_to_cart(r, theta):
    """ Converts planar Polar coordinates (radius and angle) to cartesian coordinates (x, y)
    Btw: To face outwards, theta should be the same as the rotation around the z axis"""
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return (x, y)

# To convert planar cartesian to polar coordinates
def cart_to_polar(x, y):
    """ Converts planar cartesian (x, y) coordinates to polar coordinates (r, theta)"""
    r = calc_radius(x, y)
    theta = calc_angle(x,y)
    return r, theta

# To convert planar cartesian to polar coordinates for a vector
def cart_to_polar_vec(arr):
    """ an array with 2 columns (x, y) to polar coordinates (r, theta).
    Specifically for vector cases"""
    
    r = norm(arr,axis=1)
    theta = np.arctan2(arr[:,1], arr[:,0])
    new_arr = np.column_stack((r, theta))
    return new_arr

# To flip every second row block of an array in case this is required
def flip_array(loc_arr):
    """ Method for flipping every second row block of an array"""
    
    sorted_arr = np.zeros(shape=loc_arr.shape)
    radii = np.unique(loc_arr[:,0])
    start_idx = 0
    for i, radius in enumerate(radii):
        # aidxs = np.argwhere(loc_arr[:,0] == radius)
        # arr = loc_arr[aidxs]
        arr = loc_arr[loc_arr[:,0] == radius]
        print(arr.shape)
        if (i+1) % 2 == 0:
            arr = np.flipud(arr)
            print("flipped")
        row_offs = arr.shape[0]
        sorted_arr[start_idx:(start_idx+row_offs)] = arr
        start_idx += row_offs
    return sorted_arr

# Calculates the angle at which the point sits - 
def calc_angle(x, y):
    """ Calculates the angle at which the point sits"""
    return np.arctan2(y, x)

# Calculates the distance at which a point in space is sitting
def calc_radius(x, y):
    """ Calculates the distance at which a point in space is sitting """
    return np.sqrt(x**2 + y**2)

# translates a coordinate by a certain offset - might be unneeded, with the vec_distance function
def offset_pt(x_pt, y_pt, x_offs, y_offs):
    """ Translates a coordinate point by a certain offset - in Cartesian
    returns new_x, new_y"""
    x_new = x_pt - x_offs
    y_new = y_pt - y_offs
    return x_new, y_new

# cosine similarity calculation
def cosine_sim(vec_a, vec_b):
    """ Helper function to calculate the cosine similarity.
    """    
    cos_sim = np.dot(vec_a, vec_b) / (norm(vec_a)*norm(vec_b))
    return cos_sim

# 2D - Distance calculation
def calc_distance(x, y, x_offs, y_offs):
    """ 2D- distance calculation between 2 points. Similar to the calc_radius function"""
    return np.sqrt((x-x_offs)**2 + (y-y_offs)**2)

# Loading the vector data
def load_vectors(fname, comp_list):
    """ Changed here to only take words that also exist in a pregiven list
    CAREFUL: different types of capitilisation are not discriminated!
    could potentially rerun with classes not found, but only yields ~5% more data"""

    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        if len(data)==len(comp_list):
            break
        elif tokens[0] in comp_list:
            data[tokens[0]] = tuple(tokens[1:])
            # print("Found match # {} : {}".format(len(data),tokens[0]))
    return data

# Method to truncate points
def truncate_points(point_list):
    """ Method to round an array of points to 1 decimal.
    Truncation would work by multiplication * 4, trunc, /4
    this would very likely be faster if I could preallocate the memory through initialising an empty array at the beginning"""
    point_array = []
    for point in point_list:
        pt = [point.x, point.y]
        point_array.append(pt)
    point_array = np.asarray(point_array)
    point_array = np.round(point_array, decimals=1)
    return point_array

# Function to return the length of a numpy vector
def vec_length(vec):
    """ Function to return the length of a numpy vector """
    return norm(vec)

# Function for n-dimensional distance calculation
def vector_distance(vec_a, vec_b):
    """ Function to calculate the n-dimensional distance. Subtracts vector b from vector a
    before calculating the norm """

    dist_vec = np.subtract(vec_a, vec_b)
    return norm(dist_vec)
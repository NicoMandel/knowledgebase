#!/usr/bin/env python2

import numpy as np
import pandas as pd
import rospy
import rospkg
import os.path
import yaml
from mavros_msgs.msg import PositionTarget
from geometry_msgs.msg import PoseStamped
from ml_msgs.msg import MarkerDetection
from utils import *
from knowledge_msgs.msg import PointArray, Array
from copy import deepcopy
from knowledge_msgs.msg import Object
from tf2_ros import Buffer, TransformListener, LookupException as lookE, ConnectivityException as ConnE, ExtrapolationException as ExE


class Word2Vec_Mapping:

    def __init__(self):
        """Some Schmeel about what this does"""

        # 3. something to read in the file and calculate the cosine distance
        rospack = rospkg.RosPack()
        parentDir = rospack.get_path('knowledge_server')
        self.config = yaml.load(open(os.path.join(parentDir, "config", "person_target.yaml")))
        del rospack
        # rospy.set_param("configs",self.config)


        filename = os.path.join(parentDir,"config","wiki-news-300d-1M.vec")
        data_dict = load_vectors(filename, self.config.values())
        self.target_name = self.config[0]
        target_val = data_dict[self.target_name]
        self.target_vec = np.asarray(target_val).astype(np.float)
        self.cos_df = pd.DataFrame(columns=[self.target_name], index=self.config.values())
        
        for key, value in data_dict.items():
            data_val = data_dict[key]
            data_vec = np.asarray(data_val).astype(np.float)
            data_dist = cosine_sim(self.target_vec, data_vec)
            self.cos_df.at[key,self.target_name] = data_dist

        # 1. a subscriber to the new object topic
        # This can then look up the tf
        rospy.Subscriber("newObject", Object, self.new_object_callback, queue_size=2 )
        self.mean = rospy.get_param('~mean', default=0.338)
        self.std_dev = rospy.get_param('~stddev', default=0.0606)
        self.upper_thresh = self.mean+2*self.std_dev
        self.lower_thresh = self.mean-2*self.std_dev
        # transform objects
        self.tf_buffer = Buffer(rospy.Duration(10.0))  # tf buffer length
        self.tf_listener = TransformListener(self.tf_buffer)

        # 3. A client to the camera mapping module - to find out how well the entire area is already mapped
        self.seen_points = np.zeros(shape=(1,2))
        self.polar_points = []
        rospy.Subscriber("seenPoints", PointArray, self.map_callback, queue_size=1)
        self.threshold = rospy.get_param('~pt_threshold', default=10)
        
        ang_bins = rospy.get_param('~ang_bins', default=18)
        self.abins = np.linspace(-np.pi, np.pi, ang_bins+1)
        self.rmax = rospy.get_param('~rmax', default=12)
        self.rbins = np.linspace(0,self.rmax,self.rmax+1) # gets equal stepped bins
        self.ang_bins = ang_bins
        self.path_pub = rospy.Publisher("Path", Array, queue_size = 1)


    # Callback on the topic that gets published when a new object is registered
    def new_object_callback(self, msg):
        """ Callback on the topic that gets published when a new object is registered """
        
        obj_type = msg.object_id.split("_")[0]
        cosine = self.cos_df.at[obj_type, self.target_name]
        if cosine > self.upper_thresh:      #  or cosine < self.lower_thresh - using flipup for inversing list of waypoints
            rospy.logwarn("Found Significant Object: {}, value: {}".format(
                obj_type, cosine
            ))
            new_wp_array = self.triggered_method(cosine, msg.object_id, msg.header.stamp)
        else: rospy.logwarn("Insignificant Object registered")
        
    # Callback on the seen points topic  
    def map_callback(self, msg):
        """Callback on the seen points topic """

        # Truncate the points to 1 digit after the comma
        point_array = truncate_points(msg.points)
        # Register the points
        self.register_points(point_array)
        
    # Method that compares whether points appear in both sets
    def register_points(self, point_array):
        """ Method to compare whether points appear in both sets """
        
        new_arr = np.concatenate((self.seen_points,point_array))       
        self.seen_points = np.unique(new_arr, axis=0)

    # Triggered Method
    def triggered_method(self, dist, name, stamp):
        """ Triggered method when a significant marker is detected"""
        
        # 1. look up the transfrom of the point where the detected marker is at - a static tf
        try:
            trans = self.tf_buffer.lookup_transform(
                "map",
                name,
                stamp,
                rospy.Duration(0.5) 
            )
            x = trans.transform.translation.x
            y = trans.transform.translation.y 

        except (lookE, ConnE, ExE) as e:
            rospy.logerr("Transform not possible: {}. Looking for last known UAV position".format(e))
            msg = rospy.wait_for_message("mavros/local_position/pose", PoseStamped)
            x = msg.pose.position.x 
            y = msg.pose.position.y
        
        # 2. do a histogram with the transform x,y offset
        hist = self.evaluate_points_polar(x, y)
        new_locations = self.calc_path(hist, dist)

        # 3. Reconvert the waypoints to the global waypoints               
        msg = Array()
        msg.pos_x = x
        msg.pos_y = y
        msg.r_max = self.rmax
        msg.ang_bins = self.ang_bins
        msg.rows = new_locations.shape[0]
        msg.cols = new_locations.shape[1]
        msg.data = new_locations.flatten()
        self.path_pub.publish(msg)

    # Helper function - could maybe also be thrown in utils
    def evaluate_points_polar(self, x_offs, y_offs):
        """Method that puts points in a polar histogram """

        polar_pts = []
        data = deepcopy(self.seen_points)
        for pt in data:
            # Offsetting the point to be UAV-Centred
            x_new, y_new = offset_pt(pt[0], pt[1], x_offs, y_offs)
            # Converting the points to polar
            r, ang = cart_to_polar(x_new, y_new)
            polar_pts.append(tuple([r, np.round(ang, decimals=1)]))
        
        polar_array = np.asarray(polar_pts)
        hist, _, _ = np.histogram2d(polar_array[:,1], polar_array[:,0], bins=(self.abins, self.rbins))
        return hist

    # Helper method to calculate a path from a histogram of radial data using a threshold
    def calc_path(self, hist, cosine_sim):
        """ Helper function to calculate a path from a histogram - does not flip values
        thresholds seen areas """

        # hist.T is a normalised matrix - cols = angles, rows = r
        # Omit areas that are considered well explored
        idcs = np.argwhere(hist.T < self.threshold)
        angles = self.abins[idcs[:,1]+1]
        radii = self.rbins[idcs[:,0]+1]
        # Get the boundaries of the bins and use those as new wp
        # locs = np.column_stack((angles, radii))
        x, y = polar_to_cart(radii, angles)
        new_locations = np.column_stack((x, y))
        # If the similarity is bad, flip the waypoint list upside down
        if cosine_sim < self.lower_thresh:
            new_locations = np.flipud(new_locations)
            rospy.logwarn("Negative Semantic connectedness - inversing order of waypoints")
        
        return new_locations

    
    
if __name__=="__main__":
    rospy.init_node("map_server")
    wvc = Word2Vec_Mapping()
    rospy.logwarn("Initialised Node with the Word2Vec Mapping Module")
    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except rospy.ROSInterruptException as e:
            rospy.logerr_once(e)
            
    

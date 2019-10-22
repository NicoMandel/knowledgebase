#!/usr/bin/env python2

import rospy
from geometry_msgs.msg import PoseStamped
import numpy as np           
import math
import tf2_ros
import tf2_geometry_msgs
from knowledge_server.msg import Point2D, PointArray
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo

class CameraTransform:

    # A class which transforms the points from the predefined camera ray-casting model and sends them back out
    def __init__(self, min_depth=0.1, max_depth=2.6, fov=1.047, distance_steps=6, z=1):
        """ A class which queries the tf tree periodically, 
            transforms the points from a predefined camera model and sends them back out
        """
        
        # The camera pinhole model points
        self.camera_points = self.get_camera_points(fov,min_depth,max_depth,distance_steps)
        # Transform buffer lookup
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # A publisher on the /seenPoints topic
        # Publishing the explored points topic
        self.points_pub = rospy.Publisher('seenPoints', PointArray,queue_size=1)
        self.z = z

    # A method to calculate the points which the camera can see in the UAV body frame - only called once
    def get_camera_points(self, fov, min_depth, max_depth, distance_steps):
        """ A method to calculate the points which the camera can see - in the UAV body frame, pinhole model"""
        
        # Hard-coded camera values - about 50 points 
        # This results in a hard-coded set of points, which are visible in the frame of the uav
        point_list = []
        max_angle = 0.50*fov       
        min_angle = (0.50*fov)*-1
        depth_range = max_depth-min_depth
        dist_step_size = (max_depth-min_depth)/distance_steps
        resolution = dist_step_size
        
        for dist in np.arange(min_depth, max_depth+dist_step_size/2, dist_step_size):
            # to reduce the number of points per arc
            no_of_points = np.ceil((dist * fov / resolution))  
            angle_step_size = fov/no_of_points
            if no_of_points > 1:
                angle_step_size_lower = (max_angle-min_angle)/(no_of_points-1)
                angle_step_size_upper = (max_angle-min_angle)/(no_of_points+1)
                step_sizes = (angle_step_size_lower, angle_step_size, angle_step_size_upper)
                candidates = ((angle_step_size_lower-resolution)**2, (angle_step_size-resolution)**2, (angle_step_size_upper-resolution)**2)
                angle_step_size = step_sizes[np.argmin(candidates)]

            for angle in np.arange(min_angle,max_angle+angle_step_size/2,angle_step_size):
                x = dist * math.cos(angle)     
                y = dist * math.sin(angle)
                point = (x, y, 1)
                point_list.append(point)
        
        point_array = np.asarray(point_list)
        return point_array

    # A method to cycle through the transfrom and publishing
    def do_transform_cycle(self):
        """ A method to cycle through and run the transform and publishing """
        points_world = self.transform_points(self.camera_points, self.z)
        if points_world:
            self.publish_points(points_world)
        else:
            rospy.logwarn("No points published, List object is empty, could not transform")

    # Using the tf2 transform message to look up the list of points in the world frame and sending it out
    def transform_points(self, point_array, z):
        """ Using the tf2 transform to look up the list of points in the world frame and sending it out"""
        points_world = []
        try:
            transform = self.tf_buffer.lookup_transform(
                "map",
                "base_link",
                rospy.Time(0),          # 
                rospy.Duration(0.05)    # wait for 0.05 seconds for the transform            
            )
            timestamp = rospy.Time.now()
            for point in point_array:
                pt = PointStamped()
                pt.header.frame_id = "base_link"
                pt.header.stamp = timestamp
                pt.point.x = point[0]
                pt.point.y = point[1]
                pt.point.z = z
                point_transformed = tf2_geometry_msgs.do_transform_point(point=pt, transform=transform)  
                point2D = Point2D()
                point2D.x = point_transformed.point.x
                point2D.y = point_transformed.point.y
                points_world.append(point2D)
            # send the list of points
            return points_world
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as tfe:
            rospy.loginfo(tfe)
            rospy.Rate(20).sleep()
            return None
            
    # Method to publish the array of points in the world frame
    def publish_points(self, point_array):
        """ Method to publish the array of points in the world frame"""
        self.points_pub.publish(point_array)

    ### Deprecated - self - calculated methods
    # Receives an x,y position and a yaw (in rad) and calculates them in a global view
    def transform_visible_points(self, x, y, yaw):
        """ Receives an x,y position of the UAV and uses the visible field to 
        calculate a discrete set of points that are defined as already seen - p.Corke book p. 25ff """
        self.transform_array[0,0] = math.cos(yaw)
        self.transform_array[0,1] = math.sin(yaw)*-1
        self.transform_array[1,0] = math.sin(yaw)
        self.transform_array[1,1] = math.cos(yaw)
        self.transform_array[0,2] = x
        self.transform_array[1,2] = y
        world_pts = np.matmul(self.transform_array,self.viewpoints)
        world_pts = np.around(world_pts[0:1,:],decimals=1)  # only the first two rows, but all columns
        return world_pts

    # Local position callback - puts the explored points in a set to a single digit
    def local_pos_callback(self, data):
        """ Method to receive new information about the UAVs position. Should also update the observed area """
        # euler = euler_from_quaternion(data.pose.orientation) # check whether this is correct, or whether axes sequence is required
        # # world_pts = self.transform_visible_points(data.pose.position.x, data.pose.position.y, euler[2])
        # for pt in world_pts:
        #     self.seen_points.add(pt)


class CameraTransformVertical:
    """ A class which sends out a camera transform for the case of a bottom-facing camera - pretty much a copy of the top class
    Can be adapted to a sophisticated model using the formulas described here:
    https://photo.stackexchange.com/questions/56596/how-do-i-calculate-the-ground-footprint-of-an-aerial-camera"""

    # A camera transform class for the case of a bottom-facing camera
    def __init__(self, timeout=5):
        """ A constructor for the bottom facing camera """

        self.detection_distance = rospy.get_param('~detection_threshold', default=1.8)
        self.no_of_pts = rospy.get_param('~no_of_pts', default=50)
        self.pts_frame = rospy.get_param('~pts_frame', default='base_link')
        # Camera Stuff here
        msg = rospy.wait_for_message("/iris_bottom_cam/usb_cam/camera_info", CameraInfo, timeout=timeout)
        img_height = msg.height
        img_width = msg.width
        self.focal_length = msg.K[0]        # vertical and horizontal fov are mostly the same
        # central_v = msg.K[6]
        # central_h = msg.K[3]
        self.hfov = 2*np.arctan((img_width/2)/self.focal_length)
        self.vfov = 2*np.arctan((img_height/2)/self.focal_length)
        self.aspect_ratio_inv = float(img_height)/float(img_width)
        # wfov = aspect_ratio_inv * hfov
        self.FPh_prefix = np.tan(self.hfov/2)
        self.FPv_prefix = np.tan(self.vfov/2)
        aspect_ratio = 1/self.aspect_ratio_inv
        vert_count = np.sqrt(self.no_of_pts/aspect_ratio)
        self.horiz_count = np.round(aspect_ratio*vert_count)
        self.vert_count = np.round(vert_count)
        self.vert_spacing = np.linspace(-self.vfov, self.vfov, self.vert_count)
        self.horiz_spacing = np.linspace(-self.hfov,self.hfov, self.horiz_count)
        # # The spacing for the points

        # Position stuff
        # Transform buffer lookup
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.points_pub = rospy.Publisher('seenPoints', PointArray,queue_size=1)
    
    # A method to cycle through the transfrom and publishing
    def do_transform_cycle(self):
        """ A method to cycle through and run the transform and publishing """
        points_world = self.transform_points()
        if points_world:
            self.publish_points(points_world)
        else:
            rospy.logwarn("No points published, List object is empty, could not transform")

    # Using the tf2 transform to look up the points in the world frame and send them out                
    def transform_points(self, z_offs=0.1):
        """ Using the tf2 transform to look up the list of points in the world frame and sending it out"""
        points_world = []
        try:
            transform = self.tf_buffer.lookup_transform(
                "map",
                self.pts_frame,    # Change this to be potentially the camera frame, depending on the way the points are returned
                rospy.Time(0),          # 
                rospy.Duration(0.05)    # wait for 0.05 seconds for the transform            
            )
            # Here we have to calculate the point array in the body frame depending on the height of the UAV - the transform SHOULD do the yaw
            z = transform.transform.translation.z - z_offs
            if z < self.detection_distance:
                point_array = self.get_camera_points(z=z)
                
                timestamp = rospy.Time.now()
                for point in point_array:
                    pt = PointStamped()
                    pt.header.frame_id = self.pts_frame
                    pt.header.stamp = timestamp
                    pt.point.x = point[0]
                    pt.point.y = point[1]
                    pt.point.z = z
                    point_transformed = tf2_geometry_msgs.do_transform_point(point=pt, transform=transform)  
                    point2D = Point2D()
                    point2D.x = point_transformed.point.x
                    point2D.y = point_transformed.point.y
                    points_world.append(point2D)
                # send the list of points
                return points_world
            else:
                return None
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as tfe:
            rospy.loginfo(tfe)
            rospy.Rate(20).sleep()
            return None

    # Method to publish the array of points in the world frame
    def publish_points(self, point_array):
        """ Method to publish the array of points in the world frame"""
        self.points_pub.publish(point_array)
        
    # Method to calculate the points visible in the CAMERA frame of the uav
    def get_camera_points(self, z):
        """ Method to calculate the points visible in the CAMERA frame of the UAV
        This can be improved by changing the function to rely on numpy"""
        
        # the Z value is the height of the UAV (=depth of the point)
        point_list = []
        for vert in self.vert_spacing:
            for horiz in self.horiz_spacing:
                vert_val = np.tan(vert)*z
                horiz_val = np.tan(horiz)*z
                point_list.append(tuple([vert_val, horiz_val]))
        pt_array = np.asarray(point_list)
        return pt_array

        


if __name__=="__main__":
    try:
        rospy.init_node("cam_tf")
        # cam_tf = CameraTransform()
        cam_tf=CameraTransformVertical()
    except Exception as e:
        rospy.logerr(e)
    cycle_rate = rospy.Rate(1.0)
    while not rospy.is_shutdown():
        try:
            cam_tf.do_transform_cycle()
            cycle_rate.sleep()
        except rospy.ROSInterruptException as e:
            rospy.logerr_once(e)

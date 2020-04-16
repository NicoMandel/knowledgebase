#!/usr/bin/env python2

from mavros_msgs.srv import ParamSetRequest, ParamSet, ParamSetResponse
from knowledge_msgs.msg import Array
from pymavlink import mavutil
from geometry_msgs.msg import PoseStamped, Quaternion
import rospy
import numpy as np
from utils import offset_pt, polar_to_cart, vector_distance
from tf.transformations import quaternion_from_euler
import pandas as pd
import sys

import os.path
import rospkg
from std_msgs.msg import Bool 
    
class Word2Vec_navigation:
    """ A navigation class which sends OFFBOARD commands to a UAV. See documentation above """

    def __init__(self, use_file):
        """ A navigation class"""
        
        self.dist_to_goal = 10000 # high initialisation
        self.distance_threshold = rospy.get_param('~distance_threshold', default=1.0)
        self.const_height = rospy.get_param('~const_height', default=1.5)
        self.list_ptr = 0
        try:
            self.coord_array = self.getInitialListFromFile(use_file)
            # rospy.logerr("Read target coordinates from file: {}".format(self.coord_array[:10]))
        except IOError:
            rospy.logerr("No File Found. Using default defined file from ros params")
            try:
                use_file = rospy.get_param('~file_id')
                rospack = rospkg.RosPack()
                wd = os.path.join(rospack.get_path('knowledge_server'), 'config')
                self.coord_array = self.getInitialListFromFile(os.path.join(wd, use_file))
            except KeyError:
                rospy.logerr("Param not defined. Using initial circular path")
                self.coord_array = self.getInitialList()
	
	# Shutdown publisher
	self.shutdown_pub = rospy.Publisher(shutdown_topic, Bool, queue_size=5)	
        
        rospy.Subscriber("target",PoseStamped, self.target_callback, queue_size=1)
        rospy.Subscriber("Path", Array, self.path_callback, queue_size = 2)
        self.next_wp_pub = rospy.Publisher("nextWP", PoseStamped, queue_size = 5)
        self.local_pos_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self.local_pos_callback, queue_size=2)
        first_wp = PoseStamped()
        new_target = self.getPoseStamped(self.coord_array[self.list_ptr,:],np.asarray([0.1, 0.1]))
        rate = rospy.Rate(2)
        for i in range(5):
            self.next_wp_pub.publish(new_target)
            rate.sleep()
    
    # A callback method on the local position topic
    def local_pos_callback(self, msg):
        """ A callback method on the local position topic, getting the pose of the uav"""
        vec_a = np.asarray(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
            ).astype(np.float)
        vec_b = np.asarray(
            [self.coord_array[self.list_ptr,0], self.coord_array[self.list_ptr,1], self.const_height]
            ).astype(np.float)
        self.dist_to_goal = vector_distance(vec_a, vec_b)
        # rospy.loginfo("Evaluating distance to: {}. is: {}".format(vec_b, self.dist_to_goal))
        if self.dist_to_goal < self.distance_threshold:
            self.list_ptr += 1
            self.setnextwaypoint()


    # A callback on the target topic
    def target_callback(self, msg):
        """ A Callback on the target topic """
        
        name = msg.header.frame_id
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.target_sequence(name)
    
    # A target action sequence to initiate
    def target_sequence(self, name):
        """ In accordance with the  Navigation Recommendations"""
        # and then to land - by changing the list elements
        msg =rospy.wait_for_message("/mavros/local_position/pose", PoseStamped)
        x_curr = msg.pose.position.x
        y_curr = msg.pose.position.y
        current_pos = np.asarray([x_curr, y_curr])
        rospy.logerr("Found Target: {} at: {}, {}. Landing".format(
                name, x_curr, y_curr
            ))
        new_vec = np.asarray([x_curr,y_curr])
        
        finalwp = self.getPoseStamped(self.coord_array[self.list_ptr,:],current_pos)
        self.coord_array = np.concatenate((current_pos, new_vec))
        self.list_ptr = 0
        finalwp.pose.position.z = 0.0
        self.next_wp_pub.publish(finalwp)
        self.local_pos_sub.unregister()
       
	# Shutdown hook
	final_msg = Bool()
	self.shutdown_pub.publish(final_msg)
 
    # A callback on the path topic - which updates the mission path
    def path_callback(self,msg):
        """ A callback which takes an Array msg of new waypoints  """
        try:
            x_offs = msg.pos_x
            y_offs = msg.pos_y
            arr = msg.data
            arr = np.reshape(arr, (msg.rows, msg.cols))
            self.r_max = msg.r_max
            self.ang_bins = msg.ang_bins
            
            # convert the data
            # the offset is the relative position to the origin - so adding vectors
            x_new, y_new = offset_pt(arr[:,0], arr[:,1], x_offs=x_offs*(-1), y_offs=y_offs*(-1))
            self.coord_array = np.column_stack((x_new, y_new))
            old_vec = np.array([x_offs, y_offs])
            self.coord_array = np.insert(self.coord_array, 0, old_vec, axis=0)
            self.list_ptr = 0

            # Send the waypoint out to restart the process - make sure things are right with the array pointer here!
            next_wp = self.getPoseStamped(self.coord_array[1,:],self.coord_array[0,:])
            self.next_wp_pub.publish(next_wp)

        except Exception as e:
            rospy.logerr(e)

    # A helper method to create an initial list
    def getInitialList(self, init_r = 10, angles=6):
        """ A helper method to create an initial list of waypoints to go to"""

        msg = rospy.wait_for_message('/mavros/local_position/pose', PoseStamped)
        init_x = msg.pose.position.x
        init_y = msg.pose.position.y
        somelist = [tuple([init_x, init_y])]
        abins = np.linspace(-np.pi, np.pi, angles)
        for r in range(init_r-1):
            for ang in abins:
                x, y = polar_to_cart(r+1, ang)
                somelist.append(tuple([x, y]))

        somearray = np.asarray(somelist)
        return somearray

    # Another helper method to get the initial list from the file
    def getInitialListFromFile(self, filename):
        """ A helper method to read in the file with the predefined waypoints """
        locations = pd.read_csv(filename, header=None)
        # Offset all the points by the local position
        msg = rospy.wait_for_message('/mavros/local_position/pose', PoseStamped)
        init_x = msg.pose.position.x
        init_y = msg.pose.position.y
        x_vec = locations.values[:,0]
        y_vec = locations.values[:,1]
        new_x, new_y = offset_pt(x_vec, y_vec, -init_x, -init_y)
        coord_array = np.column_stack((new_x, new_y))
        return coord_array

    # A method to set the next possible waypoint on the uav
    def setnextwaypoint(self):
        """ A method which checks if the current waypoint has been reached and then sets the new waypoint
        """
        try:
            self.list_ptr+=1
            old_vec = self.coord_array[self.list_ptr-1,:]
            new_vec = self.coord_array[self.list_ptr,:]

            rospy.loginfo("Finished with waypoint: {}, Proceeding to: {} of {}".format(
                self.list_ptr-1, self.list_ptr, self.coord_array.shape[0]
            ))
            rospy.loginfo("New Waypoint at: x: {:.2f}, y: {:.2f}".format(
                new_vec[0], new_vec[1]
            ))
            # send out the new pose as a pose_stamped
            new_target = self.getPoseStamped(new_vec,old_vec)
            rate = rospy.Rate(2)
            for i in range(5):
                self.next_wp_pub.publish(new_target)
                rate.sleep()
        except IndexError:
            rospy.logwarn("No further waypoints to explore in list. Shutting Down")
	    final_msg = Bool()
	    self.shutdown_publ.publish(final_msg)
	    

    # A method to return a PoseStamped object for two pairs of x, y coordinates
    def getPoseStamped(self, vec_new, vec_old):
        """ A method to return a PoseStamped object for two pairs of x, y coordinates """

        dist_vec = np.asarray(offset_pt(vec_new[0],vec_new[1],vec_old[0],vec_old[1]))
        yaw = np.arctan2(dist_vec[1],dist_vec[0])
        quat = quaternion_from_euler(0, 0, yaw)            
        new_target = PoseStamped()
        new_target.header.frame_id = "map"
        new_target.header.stamp = rospy.Time.now()
        new_target.pose.orientation = Quaternion(*quat)
        new_target.pose.position.x = vec_new[0]
        new_target.pose.position.y = vec_new[1]
        new_target.pose.position.z = self.const_height
        return new_target


def set_parameter(param_dict, timeout=5, rate = 2):
    """ A helper function to call before takeoff to set parameter values to decrease the horizontal velocity """

    try:
        rospy.wait_for_service("/mavros/param/get") # just use the set service as a proxy
        set_param = rospy.ServiceProxy("/mavros/param/set", ParamSet)
    except rospy.ServiceException:
        raise("Could not initialise services")
    
    sleep_rate = rospy.Rate(rate)

    for key, value in param_dict.items():
        for i in range(timeout*rate):
            try:
                req = ParamSetRequest()
                req.param_id = key
                req.value.real = float(value)
                resp = set_param(req)
                if resp.success:
                    rospy.logwarn("Successfully set: {} to: {}".format(
                        key, value
                    ))
                    break
                else:
                    raise rospy.ServiceException("Acknowledgement failed")
            except rospy.ServiceException as srvex:
                rospy.logwarn(srvex)
                sleep_rate.sleep()



if __name__=="__main__":
    rospy.init_node("navigator")
    rate = rospy.Rate(1)
    args = rospy.myargv(argv=sys.argv)
    try:
        use_file = args[1]
    except:
        rospy.logwarn("No Argument provided - continuing without predefined Waypoints")
        use_file = None
    rospy.wait_for_message('/mavros/local_position/pose', PoseStamped)    
    # Have the node sleep for a little while, before actually starting to do stuff. this is the last node
    for i in range(15):
        rate.sleep()

    service_dict = {
                'MPC_JERK_MAX':0.5,
                'MPC_ACC_HOR':0.5,      # This is the important parameter!
                'MPC_XY_P':0.15,
                'MPC_XY_VEL_MAX': 0.8,
                'MC_YAWRATE_MAX': 45.0, # in degrees/s
                'MC_YAW_P': 1.4, # 2.8 default
                'MC_YAWRATE_P': 0.1, # 0.2 default
                'MPC_TILTMAX_AIR': 12.0, # same value as for landing 
                'MPC_MAN_TILT_MAX': 12.0, #      in deg             
                'MPC_MAN_Y_MAX': 45.0, # in deg/s
                }
    set_parameter(service_dict)
    # rospack = rospkg.RosPack()
    # parentDir = rospack.get_path('knowledge_server')
    # use_file = os.path.join(parentDir, 'config', 'Initial_WPs.csv')
    navServ = Word2Vec_navigation(use_file)
    rospy.loginfo("Started Word2Vec-Nav-server_node")

    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except rospy.ROSInterruptException as e:
            rospy.logerr_once(e)
            

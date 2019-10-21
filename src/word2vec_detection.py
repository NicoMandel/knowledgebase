#!/usr/bin/env python2

import numpy as np
import pandas as pd
import rospy
import rospkg
import os.path
import yaml
from geometry_msgs.msg import PoseStamped
from ml_msgs.msg import MarkerDetection
from utils import cosine_sim, calc_distance, load_vectors
from knowledge_server.msg import Dataframe 
from knowledge_server.srv import RegisterObject, RegisterObjectRequest, RegisterObjectResponse

class Word2Vec_Detection:
    """ A class which serves as the detection of targets and registering objects"""
    
    def __init__(self):

        # something to read in the file and calculate the cosine distance
        rospack = rospkg.RosPack()
        parentDir = rospack.get_path('knowledge_server')
        self.config = yaml.load(open(os.path.join(parentDir, "config", "person_target.yaml")))
        del rospack

        # a client to the register pose server
        try:
            rospy.wait_for_service('registerStaticTf')
            self.register_pose = rospy.ServiceProxy('registerStaticTf', RegisterObject)
        except rospy.ServiceException as e:
            rospy.logerr("Could not connect to registering pose server:", e)
        # 1. a subscriber to the marker_callback topic 
        rospy.Subscriber("/ml_landmarks/detected_markers", MarkerDetection,self.marker_callback,queue_size=1)
        self.target_pub = rospy.Publisher("target", PoseStamped, latch=True, queue_size = 1)
        
        # 2. a subscriber to the known objects topic - to see how many have already been registered
        rospy.Subscriber("allObjects", Dataframe, self.df_callback, queue_size = 2 )

        # 3. loading the parameters - detection threshold
        self.col_name = rospy.get_param('count_col_name', default="count")
        self.detect_thresh = rospy.get_param('~marker_detect_thresh', default=0.6)
        self.serv_timeout = rospy.get_param("~service_timeout", default=1)
        self.serv_freq = rospy.get_param("~service_frequency", default=5)

        # Creating the dataframes
        self.registered = pd.DataFrame(
            index= list(self.config.values()),columns=[self.col_name],data=np.zeros((len(self.config), 1))
            )
        self.received = pd.DataFrame(
            index= list(self.config.values()),columns=[self.col_name],data=np.zeros((len(self.config), 1))
            )
        


    # Callback on the Marker topic
    def marker_callback(self, data):
        """ Passive callback only checking whether a marker is already registered and sending the information out """

        for marker in data.markers:
            if marker.marker_id == 0:
                # name = self.config[marker.marker_id]
                msg = PoseStamped()
                msg.pose = marker.pose
                msg.header.frame_id = self.config[marker.marker_id]
                msg.header.stamp = data.header.stamp           
                
                self.target_pub.publish(msg)
                rospy.logerr_once("Target found")

            elif self.registered.at[self.config[marker.marker_id],self.col_name] == self.received.at[self.config[marker.marker_id],self.col_name]:
                if marker.marker_confidence > self.detect_thresh:
                    registered = self.register_static_tf(marker.pose, marker.marker_id, data.header)
                    if registered:
                        self.registered.at[self.config[marker.marker_id],self.col_name] = self.registered.at[self.config[marker.marker_id], self.col_name] + 1 
                        rospy.loginfo("No errors registering new object: {}".format(
                            self.config[marker.marker_id]
                        ))
                    else:
                        rospy.logwarn("Already registered Object at this location")
                else:
                    rospy.loginfo("Seen Marker {}, confidence too low ({:.2f}) to register.".format(
                        marker.marker_id, marker.marker_confidence
                    ))

    # Callback on the dataframe topic
    def df_callback(self, msg):
        """ A callback on the dataframe topic to check the counts of registered objects"""
        # 1. unpack the incoming df
        try:
            arr = msg.data
            indices = msg.headers
            arr = np.reshape(arr, (len(indices), -1))
            incoming_df = pd.DataFrame(data=arr,
                                    index=indices,
                                    columns=[self.col_name])
            self.received = incoming_df
            # rospy.logwarn(new_df)
        except Exception as e:
            rospy.logerr("Could not convert DF coming into callback {}".format(e))
        # 2. compare it with the existing df

        # 3. Check if all the dfs are true

    # Helper Method to call the service for registering a static tf
    def register_static_tf(self, pose, marker_id, header):
        """ Helper Method to call the service for registering a static tf"""
        
        req = RegisterObjectRequest()
        req.pose.header = header
        req.pose.pose = pose
        req.name = self.config[marker_id]
        rate = rospy.Rate(self.serv_freq)
        success = False
        for i in range(self.serv_timeout*self.serv_freq):
            try:
                res = self.register_pose(req)
                if res.success:
                    success = True
                    break
                elif res.success == False:
                    break
            except rospy.ServiceException as srvex:
                rospy.logerr("could not register pose: {}".format(srvex))
            rate.sleep()
        return success
        


if __name__=="__main__":
    rospy.init_node("detection")
    wvc = Word2Vec_Detection()
    rospy.logwarn("Initialised Node with the Word2Vec Detection Module")
    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except rospy.ROSInterruptException as e:
            rospy.logerr_once(e)
            
    

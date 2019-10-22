#!/usr/bin/env python2
# HAS to be run with python2 - due to issues with tf

# And a single message from the camera_info topic - for foV and depth
# Whenever a new message is received, use the heading and the x,y 
# to discretise values in an angle up to a certain depth and register in the "explored areas"
# OR do the whole thing in a new navigation step 

import yaml
import os.path
import pandas as pd
import numpy as np
import rospy
import rospkg
from knowledge_server.srv import RegisterObject, RegisterObjectRequest, RegisterObjectResponse
from knowledge_server.msg import Object, Dataframe
from tf.transformations import quaternion_from_euler 
import tf2_ros
from tf2_ros import LookupException as lookE, ConnectivityException as ConnE, ExtrapolationException as ExE
import tf2_geometry_msgs
from geometry_msgs.msg import TransformStamped, PoseStamped
from utils import vector_distance


class TransformServer:
    """ A server to handle the transforms - which HAS to be run with python 2 due to compatibility issues of tf2 """

    def __init__(self, distance_threshold = 0.8):

        # something to read in the file and calculate the cosine distance
        rospack = rospkg.RosPack()
        parentDir = rospack.get_path('knowledge_server')
        self.config = yaml.load(open(os.path.join(parentDir, "config", "person_target.yaml")))
        del rospack

        ## Getting the parameters
        self.col_name = rospy.get_param('count_col_name', default="count")
        self.registered = pd.DataFrame(
            index= list(self.config.values()),columns=[self.col_name],data=np.zeros((len(self.config), 1))
            )

        # transform objects
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(10.0))  # tf buffer length
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer) # by default is using threading

        self.broadcaster = tf2_ros.StaticTransformBroadcaster()
        # 2. Registering a static transform 
        self.register_transform = rospy.Service('registerStaticTf', RegisterObject, self.registerStaticTf)
        # 3. Publisher for distributing a new object
        self.obj_pub = rospy.Publisher('newObject', Object, queue_size=10)
        # 4. latched publisher for distributing all the known objects
        self.known_obj = rospy.Publisher('allObjects', Dataframe, latch=True, queue_size=1)
        self.distance_threshold = rospy.get_param('~distance_threshold', default=0.8)
    
    # Function to register a static transform in the tf tree
    def registerStaticTf(self, request):
        """ Registering a static transform in the tf tree """
        
        try:
            resp = RegisterObjectResponse()
            # lookup the transform from the camera frame to the world frame
            try:
                pose_world = self.pose_cam_to_pose_world(request.pose)
                statTransStamp = self.do_transform_pose(pose_world)
            except (lookE, ConnE, ExE):
                rospy.logerr("Could not transform from board to map")
            # Convert pose to tf - the child frame id needs to be converted
            # Registering stuff done here - if the dataframe value does not get read out, but pointed to, we might need a copy statement
            name = request.name
            count = self.registered.at[name,self.col_name]
            i = 1
            do_the_registering = True
            while i <= count:
                object_id = name+"_"+str(float(i))
                # the pose that got sent, already transformed:
                new_pose = np.asarray([pose_world.pose.position.x, pose_world.pose.position.y, pose_world.pose.position.z])
                trans = self.tf_buffer.lookup_transform(
                    "map",  # target frame
                    object_id,     # source frame
                    request.pose.header.stamp,
                    rospy.Duration(0.3)
                )
                old_pose = np.asarray([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
                distance = vector_distance(new_pose, old_pose)
                if distance < self.distance_threshold:
                    do_the_registering = False
                    resp.status = "known"
                    break
                
            if do_the_registering:
                object_id = name+"_"+str(count+1)
                statTransStamp.child_frame_id = object_id
                self.broadcaster.sendTransform(statTransStamp)
                
                # Publish to the new_object topic - using the unique name and count
                self.publish_new_object(object_id, request.pose.header.stamp)
                # Publish once to the known objects topic
                self.publish_known_objects(pose_world.header.frame_id)

                # Set response to true
                rospy.loginfo("Registered Transform: {}".format(statTransStamp))
                self.registered.at[name,self.col_name] = self.registered.at[name,self.col_name] + 1
                resp.status = "new"
            else:
                rospy.logwarn_once("Already registered an object of the same name close to the knwon one")
            resp.success = True
        except rospy.ROSException as e:
            rospy.logerr("Error on the receiving side: {}".format(e)) # turn into a raise statement?
            resp.success = False
        return resp

    # A helper method looking up the transform from the camera frame to the map frame
    def pose_cam_to_pose_world(self, pose_st):
        """ A helper method to take the transform from the camera frame to the map frame """
        
        transform = self.tf_buffer.lookup_transform(
            "map",                      # target frame
            pose_st.header.frame_id,    # source frame
            pose_st.header.stamp,
            rospy.Duration(0.3)     # The tf cycles @ roughly 100 Hz
        )
        pose_world = tf2_geometry_msgs.do_transform_pose(pose_st,transform)
        return pose_world

    # A helper method to transform a PoseStamped into a TransformStamped (with some proprietary frame_ids)
    def do_transform_pose(self, pose_st):
        """ A helper method to transform a PoseStamped into a TransformStamped (with some proprietary frame_ids)"""
        # A TransformStamped Object
        statTransStamp = TransformStamped()
        # The overhead
        statTransStamp.header.stamp = pose_st.header.stamp
        statTransStamp.header.frame_id = "map"        
        # the translation values
        statTransStamp.transform.translation.x = pose_st.pose.position.x
        statTransStamp.transform.translation.y = pose_st.pose.position.y
        statTransStamp.transform.translation.z = pose_st.pose.position.z
        # the rotation values
        quat = quaternion_from_euler(
            pose_st.pose.orientation.x,
            pose_st.pose.orientation.y,
            pose_st.pose.orientation.z
            )
        statTransStamp.transform.rotation.x = quat[0]
        statTransStamp.transform.rotation.y = quat[1]
        statTransStamp.transform.rotation.z = quat[2]
        statTransStamp.transform.rotation.w = quat[3]
        return statTransStamp

    # A method, which will publish to the topic when a new object is registered as a static tf
    def publish_new_object(self, object_id, stamp):
        """ Method to publish a new object after registering it. Called once in registerStaticTf """
        # Set a trigger here to publish to the new object
        obj = Object()
        obj.header.stamp = stamp
        obj.header.frame_id = "map"
        obj.object_id = object_id
        self.obj_pub.publish(obj)

    # A method which will publish the Dataframe to the known objects topic
    def publish_known_objects(self, _id):
        """ A method which will publish the Dataframe of registered Objects to the known objects topic"""
        msg = Dataframe()
        indices = self.registered.index.tolist()
        msg.headers = indices
        values = self.registered.values
        msg.data = values.flatten().astype(np.float)
        self.known_obj.publish(msg)


if __name__=="__main__":
    serverName = "tf_server"
    rospy.init_node(serverName)
    nameserv = TransformServer()
    rospy.loginfo("Started Service Server with Name: {}".format(serverName))
    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except rospy.ROSInterruptException as e:
            rospy.logerr_once(e)
            
#!/usr/bin/env python

import rospy
from knowledge_server.msg import Array, PointArray, Object, Dataframe
import numpy as np
from geometry_msgs.msg import PoseStamped
from utils import truncate_points, cart_to_polar_vec
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.animation
import time
import pandas as pd



class surveillance:
    """ A class to monitor and display all the topics """

    def __init__(self, start):
        """ Initialisation """

        rospy.Subscriber("Path", Array, self.path_callback, queue_size=1)
        self.new_path = np.zeros((50,2))
        rospy.Subscriber("nextWP", PoseStamped, self.old_wp_callback, queue_size=1)
        self.old_path = [tuple([0.0,0.0])]
        rospy.Subscriber("seenPoints", PointArray, self.map_callback, queue_size=1)
        self.seen_points = np.zeros(shape=(1,2))
        rospy.Subscriber("newObject", Object, self.object_callback, queue_size=1)
        rospy.Subscriber("allObjects", Dataframe, self.df_callback, queue_size=1)
        self.start = start
        self.stop = None
        rospy.Subscriber("target", PoseStamped, self.target_callback, queue_size=1)


    def path_callback(self, msg):
        """ A callback function for the new path topic"""

        x = msg.pos_x
        y = msg.pos_y
        r_max = msg.r_max
        ang_bins = msg.ang_bins
        rows = msg.rows
        cols = msg.cols
        data = msg.data
        self.new_path = np.reshape(data, (rows, cols))

    def object_callback(self, msg):
        obj_type = msg.object_id.split("_")[0]
        print(obj_type)

    def old_wp_callback(self,msg):
        """ A callback function to register all the prior waypoints """
        self.old_path.append(tuple([msg.pose.position.x, msg.pose.position.y]))

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
                                    columns=["counts"])
            self.received = incoming_df
            rospy.loginfo_once(incoming_df.head())
            # rospy.logwarn(new_df)
        except Exception as e:
            rospy.logerr("Could not convert DF coming into callback {}".format(e))
    
    def object_callback(self,msg):
        pass

    def target_callback(self,msg):
        self.stop = rospy.Time.now()
        rospy.logerr("Episode took: {} s".format((self.stop-self.start).to_sec()))

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

if __name__=="__main__":
    rospy.init_node("Surveillance_Node")
    surv = surveillance(rospy.Time.now())
    limits = 10

    fig = plt.figure(figsize=(15,9))
    ax1 = fig.add_subplot(2,2,1)
    ax1.set_xlim(-limits,limits)
    ax1.set_ylim(-limits,limits)
    ax1.title.set_text("Map - centred cartesian explored points")
    ax2 = fig.add_subplot(2,2,2)
    ax2.title.set_text("New path points")
    ax2.set_xlim(-limits,limits)
    ax2.set_ylim(-limits,limits)
    ax3 = fig.add_subplot(2,2,3)
    ax3.set_xlim(-limits,limits)
    ax3.set_ylim(-limits,limits)
    ax3.title.set_text("Old UAV Waypoints")
    ax4 = fig.add_subplot(2,2,4, projection='polar')
    ax4.title.set_text("World-Centred polar explored area")

    ang_bins = 24
    rmax = 10
    abins = np.linspace(-np.pi, np.pi, ang_bins+1)
    rbins = np.linspace(0,rmax,rmax+1)
    A, R = np.meshgrid(rbins, abins)

    # Animation funciton
    def animate(i):
        start = time.time()
        seen_points = deepcopy(surv.seen_points)
        old_path = deepcopy(surv.old_path)
        new_path = deepcopy(surv.new_path)
        if np.any(seen_points):
            # convert to polar
            polar_arr = cart_to_polar_vec(seen_points)
            # shove through the histogram2d
            hist, _, _  = np.histogram2d(polar_arr[:,0], polar_arr[:,1], bins=(rbins, abins))
            ax4.clear()
            pc = ax4.pcolormesh(R, A, hist.T)
            ax4.title.set_text("UAV - centred polar explored points.")

            # for the cartesian points
            ax1.clear()
            ax1.scatter(seen_points[:,0], seen_points[:,1])
            ax1.set_xlim(-limits,limits)
            ax1.set_ylim(-limits,limits)
            ax1.title.set_text("Map - centred cartesian explored points")
            
            # for the new path
            ax2.clear()
            ax2.plot(new_path[:,0], new_path[:,1], 'xr-')
            ax2.title.set_text("New path points")
            ax2.set_xlim(-limits,limits)
            ax2.set_ylim(-limits,limits)

            # for the old path
            ax3.clear()
            old_p = np.asarray(old_path)
            ax3.plot(old_p[:,0],old_p[:,1],'.b-')
            ax3.set_xlim(-limits,limits)
            ax3.set_ylim(-limits,limits)
            ax3.title.set_text("Old UAV Waypoints")     
        
        end = time.time()
        fig.suptitle("Entire process took: {} ms".format(np.round((end-start)*1000)))

    ani = matplotlib.animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()     

    while not rospy.is_shutdown():
        try:
            rospy.spin()
        except rospy.ROSInterruptException as e:
            rospy.logerr_once(e)
            

#!/usr/bin/env python2

# Documentation on the spawn model service:
# http://docs.ros.org/jade/api/gazebo_msgs/html/srv/SpawnModel.html
# following this example:
# https://answers.ros.org/question/246419/gazebo-spawn_model-from-py-source-code/
import rospy
from gazebo_msgs.srv import DeleteModel, SpawnModel, SpawnModelRequest, SpawnModelResponse, GetModelState
from gazebo_msgs.msg import ModelState
from tf.transformations import quaternion_from_euler
from geometry_msgs.msg import Quaternion, Pose, Point
import os.path
import rospkg
from os import walk
import numpy as np
from utils import polar_to_cart
# from scipy.stats import invgauss
import pandas as pd
import matplotlib.pyplot as plt
import sys

class SimulationConfig:

    def __init__(self, _id, target, good_obj, bad_obj, rand_obj):
        self.id = _id
        self.target = target
        self.good_obj = good_obj
        self.bad_obj = bad_obj
        self.rand_obj = rand_obj

    def __repr__(self):
        return str(self.id)

    def __str__(self):
        return str(self.id)
    

def getLocation(model_name, service):
    """ Calls the Model State Service from Gazebo to get x,y coordinates of objects"""
    model_coordinates = service(model_name,"")
    if model_coordinates.success:
        arr = np.asarray([model_coordinates.pose.position.x,  model_coordinates.pose.position.y])
        return arr
    else:
        return None

def callSpawnService(service, model_number, x, y, parentdir, prefix="board_"):
    default_z = 0.1
    obj_pose = Pose()
    obj_pose.position.x = x
    obj_pose.position.y = y
    obj_pose.position.z = default_z
    orient = Quaternion(*quaternion_from_euler(0, 1.57, 0)) # might be able to change the yaw here
    obj_pose.orientation = orient
    # # service call arguments: model_name model_xml robot_namespace initial_pose reference_frame
    with open(os.path.join(parentdir,prefix+str(model_number),"model.sdf"), "r") as f:
        product_xml = f.read()
    req = SpawnModelRequest()
    req.initial_pose = obj_pose
    req.model_name = prefix+str(model_number)
    req.model_xml = product_xml
    req.robot_namespace = ""
    req.reference_frame = "map"    # Empty for world
    try:
        service(req)
    except rospy.ServiceException:
        raise

### Helper Rest for storing the stuff I did on the inverse Gaussian - before throwing it out
def somehelperstuff():
    max_r = 6  # Actually half of the maximum search radius
    goal_idx = [0]
    pos_idx = [1, 2]
    neg_idx = [19, 20]
    donotcare = list(np.arange(21))
    locations_df = pd.DataFrame(
        columns=['x','y'], index=donotcare
        )
    # interesting.extend(goal_idx)
    # interesting = goal_idx.extend(neg_idx).extend(pos_idx)
    donotcarelist = list(set(donotcare) - set(goal_idx+pos_idx+neg_idx))    
        # first: set the goal object somewhere on the outside rim
    goal_angle = np.random.uniform(-np.pi, np.pi, size=1)
    x_goal, y_goal = polar_to_cart(max_r, goal_angle)
    lam = 0.2
    mu = 1.0
    scale = lam
    muu = mu/lam
    for elem in pos_idx:
        rad = invgauss.rvs(muu, scale=scale, size = 1)
        ang = np.random.normal(goal_angle-np.pi, np.pi/2, 1)
        x, y = polar_to_cart(max_r-rad,ang) # here is where the radius is subtracted
        # send out the spawning stuff here
        # callSpawnService(spawn_model, elem, x, y, parentdir, prefix)
        locations_df.at[elem,'x'] = x
        locations_df.at[elem,'y'] = y
        
        

    # Negative indices stuff here
    lam = 1/lam
    scale = lam
    muu = mu/lam
    for elem in neg_idx:
        rad = invgauss.rvs(muu, scale=scale, size = 1)
        ang = np.random.normal(goal_angle-np.pi, np.pi, 1)  # different std.dev here!
        x, y = polar_to_cart(rad,ang)
        locations_df.at[elem,'x'] = x
        locations_df.at[elem,'y'] = y
        # send out the spawning stuff here

    # Irrelevant objects stuff here
    for elem in donotcarelist:
        rad = np.random.uniform(-np.pi, np.pi, size=1)
        ang = np.random.uniform(0, max_r, size=1)
        x, y = polar_to_cart(rad,ang)
        locations_df.at[elem,'x'] = x
        locations_df.at[elem,'y'] = y

def plotsomestuff(simulation_list, simulation_dict):
    # To have 9 simulation cases for plotting
    simulation_list.pop(-1)
    n_ax = 3
    fig =plt.figure()
    ctr = 0
    markersize = 200
    fontsize = 14
    x_limits = [-11, 11]
    y_limits = [-11, 11]
    for i in range(n_ax):
        for j in range(n_ax):
            ax = fig.add_subplot(n_ax,n_ax,i * n_ax + j+1)
            # ax.grid(True)
            sim_id = simulation_list[ctr]
            sim = simulation_dict[sim_id]
            ax.scatter(sim.rand_obj.loc[:,0], sim.rand_obj.loc[:,1],
            color='black', marker='.', s=markersize, label='Random')
            ax.scatter(sim.bad_obj.loc[:,0], sim.bad_obj.loc[:,1],
            color='red', marker='*', s=markersize, label='Negative')
            ax.scatter(sim.good_obj.loc[:,0], sim.good_obj.loc[:,1],
            color='blue', marker='P', s=markersize, label='Positive')
            ax.scatter(sim.target[0],sim.target[1],
            color='green', marker='X', s=markersize, label='Target')
            ax.set(xlim=x_limits,ylim=y_limits)
            # ax.legend()
            ax.set_title("Simulation: {}".format(sim_id), fontsize=fontsize)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ctr+=1
    plt.legend()
    plt.show()

if __name__=="__main__":
    args = rospy.myargv(argv=sys.argv)
    try:
        spawn_id = int(args[1])
        if spawn_id > 9 or spawn_id < 0:
            rospy.logwarn("Outside of bounds. Must be between 0 and 9 Continuing with 0")
            spawn_id = 0
    except:
        rospy.logwarn("No or wrong file number provided, proceeding with 0")
        spawn_id = 0


    rospy.init_node("Spawn_node",anonymous=True)   
    try:
        rospy.wait_for_service("/gazebo/spawn_sdf_model")
        rospy.wait_for_service("/gazebo/delete_model")
        rospy.wait_for_service('/gazebo/get_model_state')

        delete_model = rospy.ServiceProxy("/gazebo/delete_model", DeleteModel)
        spawn_model = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
        dist_client = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
    except rospy.ServiceException as srvex:
        rospy.logerr(srvex)

    # Get the different models here
    parentdir = os.path.join("/home","nicolas",".gazebo","models")
    prefix = "board_"
    dirs = []
    for (dirpath, dirnames, _) in walk(parentdir):
        for dirname in dirnames:
            if prefix in dirname:
                dirs.append(os.path.join(dirpath, dirname))
        break

    
    # board_0 is the goal
    # board_1 and board_2, are the positive examples
    # board_19, 20 are the negative examples

    # Get a Dataframe to store the information
    
    # Every board has the same orientation(for now - might have to change yaw)
    

    # Read the locations from the config/sims folder
    rospack = rospkg.RosPack()
    sim_parentDir = rospack.get_path('knowledge_server')
    sims_dir = os.path.join(sim_parentDir, 'config', 'sims')

    # Continue here
    simulation_list = []
    simulation_dict = {}
    for root, dirs, files in os.walk(sims_dir):
        for f in files:
            if f.endswith(".csv"):
                file_id = f.split("_")[-1].split(".")[0]
                dets = pd.read_csv(os.path.join(root,f),header=None)
                target = dets.loc[0,:].copy()
                good_obj = dets.loc[1:2,:].copy()
                bad_obj = dets.loc[3:4,:].copy()
                rand_obj = dets.iloc[-16:].copy()
                sim = SimulationConfig(file_id, target, good_obj, bad_obj, rand_obj)
                simulation_dict[file_id] = sim
                simulation_list.append(file_id) 

    # plotsomestuff(simulation_list, simulation_dict)
    # print("Test Done")

    # Can take a single simulation and call the spawning service with this
    sim_id = simulation_list[spawn_id]
    sim = simulation_dict[sim_id]
    rospy.logerr("Simulation case: {}".format(sim_id))
    for i in range(21):
        if i == 0:
            # Case: Target
            x = dets.loc[i,0]
            y = dets.loc[i,1]
            callSpawnService(spawn_model, 0, x, y, parentdir)
        elif i < 3:
            # Case: Good Objects
            x = dets.loc[i,0]
            y = dets.loc[i,1]
            callSpawnService(spawn_model, i, x, y, parentdir)
        elif i < 5:
            # Case: Bad Objects
            x = dets.loc[i,0]
            y = dets.loc[i,1]
            callSpawnService(spawn_model, 20-(i-3), x, y, parentdir)
        else:
            x = dets.loc[i,0]
            y = dets.loc[i,1]
            callSpawnService(spawn_model, i-2, x, y, parentdir)

    # print("Test Done")

    # # call here
    
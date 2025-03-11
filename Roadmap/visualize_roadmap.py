import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml
import sys
import os
import argparse
from matplotlib.patches import Rectangle
from xml.dom import minidom
from pathlib import Path

def quat2euler(quat):
    # Function that converts a quaternion [w,x,y,z] to euler angles [roll,pitch,yaw]
    # Source: https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    # Assumes that the quaternion is normalized
    w,x,y,z = quat
    roll = np.arctan2(2*(w*x+y*z),1-2*(x**2+y**2))
    pitch = np.arcsin(2*(w*y-z*x))
    yaw = np.arctan2(2*(w*z+x*y),1-2*(y**2+z**2))
    return yaw

argParser = argparse.ArgumentParser()
argParser.add_argument("-c", "--connection", help="connection type: trajs, edges, paths", default="trajs")
argParser.add_argument("-e", "--environment", help="environment file in resources/input_files/environments/", default="bar.yaml")
argParser.add_argument("-o", "--outfile", help="target file for visualization", default='foo.png')
argParser.add_argument("-d", "--directory", help="directory where roadmap files can be found (in out/)", default="latest")


args = argParser.parse_args()

mode = args.connection
# robot_dims = [1.1,0.842]
robot_dims = [.9,0.6]
diag_len = 0.25 * np.sqrt(robot_dims[0]**2 + robot_dims[1]**2)

environment_file = Path(os.path.dirname(__file__)).parent / "gym_envs" / "env_geometries" / args.environment
print(environment_file)
env_file_type= "yaml" #"yaml"

#plt.figure(figsize=(9,9))
plt.figure(figsize=(15,9))


with open(environment_file, 'r') as stream:
    try:
        env_params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


obstacles = env_params["environment"]["geometries"]
for obstacle in obstacles:
    box_center = obstacle["config"]["position"][:2]
    box_dims = obstacle["collision_geometry"]["dims"][:2]
    rect = Rectangle((box_center[0]-box_dims[0]/2.0,box_center[1]-box_dims[1]/2.0),box_dims[0],box_dims[1],
    linewidth=1,edgecolor='r',facecolor='r')
    plt.gca().add_patch(rect)
plt.xlim(-11,11)
plt.ylim(-11,11)

roadmap_dir = Path(os.path.dirname(__file__)) / "roadmap_files" / args.directory
print(roadmap_dir)
for fname in os.listdir(roadmap_dir/"trajs"):
    if fname.endswith(".txt"):
        if mode == "trajs" and fname.startswith("traj"):
            traj = np.loadtxt(roadmap_dir/"trajs"/fname,delimiter=",")
            if len(traj) == 0: continue
            if isinstance(traj[0], np.float64 ) : continue
            plt.plot(traj[:,0],traj[:,1],color='black')
            # Plot an arrow in the middle of the trajectory
            mid = int(len(traj)/2)
            plt.arrow(traj[mid,0],traj[mid,1],traj[mid+1,0]-traj[mid,0],traj[mid+1,1]-traj[mid,1],color='black',width=0.1, zorder=12)

        if mode == "paths" and fname.startswith("path"):
            traj = np.loadtxt(roadmap_dir+fname,delimiter=",")
            if len(traj) == 0: continue
            if isinstance(traj[0], np.float64 ) : continue
            plt.plot(traj[:,0],traj[:,1],color='black')
            
            i = 0
            for v in traj:
                # rectangle_corner = np.array([v[0]-diag_len*np.cos(0.25*np.pi+v[2]),
                #                     v[1]-diag_len*np.sin(0.25*np.pi+v[2])])
                # rect = Rectangle((rectangle_corner[0],rectangle_corner[1]),
                #     robot_dims[0],robot_dims[1],
                #     edgecolor='purple',facecolor='purple',angle=180.*v[2]/np.pi)
                # plt.gca().add_patch(rect)

                arrow1 = mpatches.FancyArrow(v[0],v[1],0.5*np.cos(v[2]-.5),0.5*np.sin(v[2]-.5), length_includes_head=True, color='purple', width = 0.05, zorder=10)
                arrow2 = mpatches.FancyArrow(v[0],v[1],0.5*np.cos(v[2]+.5),0.5*np.sin(v[2]+.5), length_includes_head=True, color='purple', width = 0.05, zorder=10)
                circle = mpatches.Circle((v[0],v[1]),radius = 0.5, color = 'xkcd:light purple', zorder = 9)
                plt.gca().add_patch(arrow1)
                plt.gca().add_patch(arrow2)
                plt.gca().add_patch(circle)
                plt.annotate(i,(v[0],v[1]), (v[0]+0.5,v[1]+0.5), zorder = 11)
                i+=1
            


if(mode != "paths"):
    vertices_raw = np.loadtxt(roadmap_dir / "nodes.txt", delimiter=",")

    vertices = {}
    for i in range(vertices_raw.shape[0]):
        vertices[int(vertices_raw[i,0])] = vertices_raw[i,1:]

    # plt.scatter(vertices_raw[:,1],vertices_raw[:,2],marker='o',s=100)
    for k,v in vertices.items():
        arrow1 = mpatches.FancyArrow(v[0],v[1],0.5*np.cos(v[2]-.5),0.5*np.sin(v[2]-.5), length_includes_head=True, color='purple', width = 0.05, zorder=10)
        arrow2 = mpatches.FancyArrow(v[0],v[1],0.5*np.cos(v[2]+.5),0.5*np.sin(v[2]+.5), length_includes_head=True, color='purple', width = 0.05, zorder=10)
        circle = mpatches.Circle((v[0],v[1]),radius = 0.5, color = 'xkcd:light purple', zorder = 9)
        plt.gca().add_patch(arrow1)
        plt.gca().add_patch(arrow2)
        plt.gca().add_patch(circle)

    #Label vertices with their idx
    for i in range(vertices_raw.shape[0]):
            plt.annotate(str(int(vertices_raw[i,0])),(vertices_raw[i,1],vertices_raw[i,2]),(vertices_raw[i,1]+.5,vertices_raw[i,2]+.5), zorder = 11)



if mode == "edges":
    edges_fname = roadmap_dir+"edges.txt"
    edges = np.loadtxt(edges_fname,delimiter=",")
    for edge in edges:
        vertex_from = vertices[int(edge[0])]
        vertex_to = vertices[int(edge[1])]

        # Plot arrow from vertex_from to vertex_to
        # If edge[2] == 0, plot as a dotted line.
        if edge[2] == 0:
            plt.arrow(vertex_from[0],vertex_from[1],vertex_to[0]-vertex_from[0],vertex_to[1]-vertex_from[1],linestyle='dotted',
            head_width=0.25, head_length=0.25,color='red')
        else:
            plt.arrow(vertex_from[0],vertex_from[1],vertex_to[0]-vertex_from[0],vertex_to[1]-vertex_from[1],
            head_width=0.25, head_length=0.25, fc='k', ec='k')



plt.savefig(args.outfile)
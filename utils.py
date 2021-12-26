import numpy as np
import torch
import random

import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R

import trimesh
from body_visualizer.tools.vis_tools import colors
from human_body_prior.tools.omni_tools import copy2cpu as c2c

def plot_mesh(fig, ax, mv, v, faces, num_verts, plot_stride=5):
    plt.ion()

    fig.show()
    fig.canvas.draw()

    for i in range(0, len(v), plot_stride):
        body_mesh = trimesh.Trimesh(vertices=c2c(v[i]), 
                                    faces=faces, 
                                    vertex_colors=np.tile(colors['grey'], (num_verts, 1)))

        body_mesh.apply_transform(trimesh.transformations.rotation_matrix(30, (1, 0, 0)))

        mv.set_static_meshes([body_mesh])
        body_image_wfingers = mv.render(render_wireframe=False)

        ax.clear()
        ax.imshow(body_image_wfingers)
        fig.canvas.draw()
        
        
def plot_xyz(fig, ax, xyz_ptr, plot_stride=5):
    plt.ion()

    fig.show()
    fig.canvas.draw()

    for i in range(0, len(xyz_ptr), plot_stride):
        ax.clear()

        ax.plot(xyz_ptr[i, :, 0], xyz_ptr[i, :, 1], xyz_ptr[i, :, 2], '.')
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_zlim([0, 2])

        fig.canvas.draw()
        
        
def transform(pose, trans, coef=5):
    # get shape of pose
    T, D = pose.shape

    # cut front randomly
    if np.random.rand() > 0.5:
        cut_limit = T//coef
        cut_length = np.random.randint(0, cut_limit+1)
        pose = pose[cut_length:]
        trans = trans[cut_length:]
      
    # cut last randomly
    if np.random.rand() > 0.5:
        T = pose.shape[0]
        cut_limit = T//coef
        cut_length = np.random.randint(0, cut_limit+1)
        if cut_length > 0:
            pose = pose[:-cut_length]
            trans = trans[:-cut_length]
        
    # get shape of pose, and where/how much to cut.
    T, D = pose.shape
    cut_limit = T//coef
    cut_length = np.random.randint(cut_limit//2, cut_limit+1)
    cut_index = np.random.randint(0, T-cut_length)
    
    # change pose to euler angle of T N 3 (N is the number of joints)
    euler = R.from_rotvec(pose.reshape(T*D//3, 3)).as_euler('xyz', degrees=True).reshape(T, D//3, 3)
    euler_trans = torch.Tensor(np.concatenate([euler, trans[:, np.newaxis, :]], axis=1)) # T N+1 3

    prob = np.random.rand()
    
    if prob < 0.5 and cut_length >= 2:
        # choose in the middle -> shorten it with random factor between 0.5~1.0
        scale_factor = 0.25+0.75*np.random.rand()
        
        before_cut = euler_trans[:cut_index]
        after_cut = euler_trans[(cut_index+cut_length):]
        target = euler_trans[cut_index:(cut_index+cut_length)]
        interpolated = torch.nn.functional.interpolate(target.permute(1, 2, 0), 
                                                       scale_factor=scale_factor, 
                                                       mode=random.choice(['linear'])).permute(2, 0, 1)
        euler_trans = torch.cat([before_cut, interpolated, after_cut], dim=0)
    elif 0.5 <= prob and cut_length >= 1:
        # choose in the middle -> lengthen it with random factor between 1.0~2.0
        scale_factor = 1+2*np.random.rand()
        
        before_cut = euler_trans[:cut_index]
        after_cut = euler_trans[(cut_index+cut_length):]
        target = euler_trans[cut_index:(cut_index+cut_length)]
        interpolated = torch.nn.functional.interpolate(target.permute(1, 2, 0), 
                                                       scale_factor=scale_factor, 
                                                       mode=random.choice(['linear'])).permute(2, 0, 1)
        euler_trans = torch.cat([before_cut, interpolated, after_cut], dim=0)
    
    T, N, D = euler_trans.shape
    euler = euler_trans[:, :(N-1), :].reshape(T*(N-1), D)
    trans = euler_trans[:, -1, :]
    mod_pose = R.from_euler('xyz', euler, degrees=True).as_rotvec()
    return mod_pose.reshape(T, (N-1)*D), trans
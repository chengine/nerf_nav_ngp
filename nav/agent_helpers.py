import os
import numpy as np
import torch
import json
import cv2
import time
import imageio
from nav.math_utils import vec_to_rot_matrix, rot_matrix_to_vec, rot_x, skew_matrix_torch
import subprocess

# #Helper functions
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# rot_x = lambda phi: torch.tensor([
#         [1., 0., 0.],
#         [0., torch.cos(phi), -torch.sin(phi)],
#         [0., torch.sin(phi), torch.cos(phi)]], dtype=torch.float32, device=device)

# def skew_matrix_torch(vector):  # vector to skewsym. matrix

#     ss_matrix = torch.zeros((3,3))
#     ss_matrix[0, 1] = -vector[2]
#     ss_matrix[0, 2] = vector[1]
#     ss_matrix[1, 0] = vector[2]
#     ss_matrix[1, 2] = -vector[0]
#     ss_matrix[2, 0] = -vector[1]
#     ss_matrix[2, 1] = vector[0]

#     return ss_matrix

def add_noise_to_state(state, noise):
    return state + noise

class Agent():
    def __init__(self, agent_cfg, camera_cfg, blender_cfg) -> None:

        #Initialize camera params
        self.path = camera_cfg['path']
        self.half_res = camera_cfg['half_res']
        self.white_bg = camera_cfg['white_bg']

        self.data = {
        'pose': None,
        'res_x': camera_cfg['res_x'],           # x resolution
        'res_y': camera_cfg['res_y'],           # y resolution
        'trans': camera_cfg['trans'],     # Boolean
        'mode': camera_cfg['mode']             # Must be either 'RGB' or 'RGBA'
        }   

        self.blend = blender_cfg['blend_path']
        self.blend_script = blender_cfg['script_path']

        self.iter = 0

        #Initialized pose and agent params
        self.x = agent_cfg['x0']
        self.dt = agent_cfg['dt']
        self.g = agent_cfg['g']
        self.mass = agent_cfg['mass']
        self.I = agent_cfg['I']
        self.invI = torch.inverse(self.I)

        self.states_history = [self.x.clone().cpu().detach().numpy().tolist()]

    def step(self, action, noise=None):
        #DYANMICS FUNCTION

        action = action.reshape(-1)

        newstate = self.drone_dynamics(self.x, action)

        if noise is not None:
            newstate_noise = add_noise_to_state(newstate, noise)
        else:
            newstate_noise = newstate

        self.x = newstate_noise

        new_state = newstate_noise.clone().detach()

        ### IMPORTANT: ACCOUNT FOR CAMERA ORIENTATION WRT DRONE ORIENTATION
        new_pose = torch.eye(4)
        new_pose[:3, :3] = rot_x(torch.tensor(np.pi/2)) @ vec_to_rot_matrix(new_state[6:9])
        new_pose[:3, 3] = new_state[:3]

        # Write a transform file and receive an image from Blender
        # Modify data dictionary to update pose
        self.data['pose'] = new_pose.tolist()

        # Capture image
        img = self.get_img(self.data)
        img = torch.from_numpy(img)
        self.states_history.append(self.x.clone().cpu().detach().numpy().tolist())
        self.iter += 1

        # Revert camera pose to be in body frame
        new_pose[:3, :3] = rot_x(torch.tensor(-np.pi/2)) @ new_pose[:3, :3]

        return new_pose.cpu().numpy(), new_state.cpu().numpy(), img

    def state2image(self, state):
        # Directly update the stored state and receive the image
        self.x = state

        new_state = state.clone().cpu().detach().numpy()

        new_pose = np.eye(4)
        new_pose[:3, :3] = rot_x(torch.tensor(np.pi/2)) @ vec_to_rot_matrix(new_state[6:9])
        new_pose[:3, 3] = new_state[:3]

        # Write a transform file and receive an image from Blender
        # Write a transform file and receive an image from Blender
        # Modify data dictionary to update pose
        self.data['pose'] = new_pose.tolist()

        # Capture image
        img = self.get_img(self.data)
        img = torch.from_numpy(img)
        self.img = img
        self.states_history.append(self.x.clone().cpu().detach().numpy().tolist())

        return new_pose, new_state, img

    def drone_dynamics(self, state, action):
        #State is 18 dimensional [pos(3), vel(3), R (3), omega(3)] where pos, vel are in the world frame, R is the rotation from points in the body frame to world frame
        # and omega are angular rates in the body frame
        next_state = torch.zeros(12)

        #print(state, state.shape)
        #Actions are [total thrust, torque x, torque y, torque z]
        fz = action[0]
        tau = action[1:]

        #Define state vector
        pos = state[0:3]
        v   = state[3:6]
        R_flat = state[6:9]
        R = vec_to_rot_matrix(R_flat)
        omega = state[9:]

        # The acceleration
        sum_action = torch.zeros(3)
        sum_action[2] = fz

        dv = (torch.tensor([0.,0.,-self.mass*self.g]) + R @ sum_action)/self.mass

        # The angular accelerations
        domega = self.invI @ (tau - torch.cross(omega, self.I @ omega))

        # Propagate rotation matrix using exponential map of the angle displacements
        angle = omega*self.dt
        theta = torch.norm(angle, p=2)
        if theta == 0:
            exp_i = torch.eye(3)
        else:
            exp_i = torch.eye(3)
            angle_norm = angle / theta
            K = skew_matrix_torch(angle_norm)

            exp_i = torch.eye(3) + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.matmul(K, K)

        next_R = R @ exp_i

        next_state[0:3] = pos + v * self.dt
        next_state[3:6] = v + dv * self.dt

        next_state[6:9] = rot_matrix_to_vec(next_R)

        next_state[9:] = omega + domega * self.dt

        return next_state

    def get_img(self, data):
        pose_path = self.path + f'/{self.iter}.json'
        img_path = self.path + f'/{self.iter}.png'

        try: 
            with open(pose_path,"w+") as f:
                json.dump(data, f, indent=4)
        except Exception as err:
            print(f"Unexpected {err}, {type(err)}")
            raise

        # Run the capture image script in headless blender
        subprocess.run(['blender', '-b', self.blend, '-P', self.blend_script, '--', pose_path, img_path])

        try: 
            img = imageio.imread(img_path)
        except Exception as err:
            print(f"Unexpected {err}, {type(err)}")
            raise

        img = (np.array(img) / 255.0).astype(np.float32)
        if self.half_res is True:
            width = int(img.shape[1]//2)
            height = int(img.shape[0]//2)
            dim = (width, height)
  
            # resize image
            img = cv2.resize(img, dim)

        if self.white_bg is True:
            img = img[..., :3] * img[..., -1:] + (1.0 - img[..., -1:])

        img = (np.array(img) * 255.).astype(np.uint8)
        print('Received updated image')
        return img

    def save_data(self, filename):
        true_states = {}
        true_states['true_states'] = self.states_history
        with open(filename,"w+") as f:
            json.dump(true_states, f)
        return
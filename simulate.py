import os, sys
import numpy as np
import torch
import shutil
import pathlib
import subprocess
from tqdm import trange
import argparse
from nerf.utils import *
from nerf.provider import NeRFDataset

# Import Helper Classes
from nav import (Estimator, Agent, Planner, vec_to_rot_matrix, rot_matrix_to_vec)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####################### MAIN LOOP ##########################################
def simulate(planner_cfg, agent_cfg, filter_cfg, extra_cfg, density_fn, render_fn, get_rays_fn):
    '''
    Main loop that iterates between planning and estimation.
    '''

    start_state = planner_cfg['start_state']
    end_state = planner_cfg['end_state']
    
    # Creates a workspace to hold all the trajectory data
    basefolder = "paths" / pathlib.Path(planner_cfg['exp_name'])
    if basefolder.exists():
        print(basefolder, "already exists!")
        if input("Clear it before continuing? [y/N]:").lower() == "y":
            shutil.rmtree(basefolder)
    basefolder.mkdir()
    (basefolder / "init_poses").mkdir()
    (basefolder / "init_costs").mkdir()
    (basefolder / "replan_poses").mkdir()
    (basefolder / "replan_costs").mkdir()
    (basefolder / "estimator_data").mkdir()
    print("created", basefolder)
  
    # Initialize Planner
    traj = Planner(start_state, end_state, planner_cfg, density_fn)

    traj.basefolder = basefolder

    # Create a coarse trajectory to initialize the planner by using A*. 
    traj.a_star_init()

    # From the A* initialization, perform gradient descent on the flat states of agent to get a trajectory
    # that minimizes collision and control effort.
    traj.learn_init()

    # Visualize initial trajectory
    # window = np.array([[-2., 2],
    # [-2., 2],
    # [.1, 0.5]])
    # pc = density_to_pc(density_fn, threshold=100, N=1000000, window=window)

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # pc = pc.detach().cpu().numpy()
    # X, Y, Z = pc[:, 0], pc[:, 1], pc[:, 2]
    # ax.scatter3D(X, Y, Z)

    # pos = traj.get_full_states()
    # pos = pos.detach().cpu().numpy()
    # x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    # ax.plot3D(x, y, z, 'black')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    # plt.show()

    #Change start state from 18-vector (with rotation as a rotation matrix) to 12 vector (with rotation as a rotation vector)
    start_state = torch.cat([start_state[:6], rot_matrix_to_vec(start_state[6:15].reshape((3, 3))), start_state[15:]], dim=-1).cuda()

    agent_cfg['x0'] = start_state
    # Initialize the agent. Evolves the agent with time and interacts with the simulator (Blender) to get observations.
    agent = Agent(agent_cfg, camera_cfg, blender_cfg)

    # State estimator. Takes the observations from Agent class and performs filtering to get a state estimate (12-vector)
    filter = Estimator(filter_cfg, agent, start_state, get_rays_fn=get_rays_fn, render_fn=render_fn)
    filter.basefolder = basefolder

    true_states = start_state.cpu().detach().numpy()

    steps = traj.get_actions().shape[0]

    noise_std = extra_cfg['mpc_noise_std']
    noise_mean = extra_cfg['mpc_noise_mean']

    try:
        for iter in trange(steps):
            # In MPC style, take the next action recommended from the planner
            if iter < steps - 5:
                action = traj.get_next_action().clone().detach()
            else:
                action = traj.get_actions()[iter - steps + 5, :]

            noise = torch.normal(noise_mean, noise_std)

            # Have the agent perform the recommended action, subject to noise. true_pose, true_state are here
            # for simulation purposes in order to benchmark performance. They are the true state of the agent
            # subjected to noise. gt_img is the observation.
            true_pose, true_state, gt_img = agent.step(action, noise=noise)
            true_states = np.vstack((true_states, true_state))

            # Given the planner's recommended action and the observation, perform state estimation. true_pose
            # is here only to benchmark performance. 
            state_est = filter.estimate_state(gt_img, true_pose, action)

            if iter < steps - 5:
                #state estimate is 12-vector. Transform to 18-vector
                state_est = torch.cat([state_est[:6], vec_to_rot_matrix(state_est[6:9]).reshape(-1), state_est[9:]], dim=-1)

                # Let the planner know where the agent is estimated to be
                traj.update_state(state_est)

                # Replan from the state estimate
                traj.learn_update(iter)
        return

    except KeyboardInterrupt:
        return

####################### END OF MAIN LOOP ##########################################

if __name__ == "__main__":

    ### ------ TORCH-NGP SPECIFIC ----- ###
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--iters', type=int, default=30000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

    opt = parser.parse_args()

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.empty_cache()

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = False
        opt.preload = False

    if opt.ff:
        opt.fp16 = False
        assert opt.bg_radius <= 0, "background model is not implemented for --ff"
        from nerf.network_ff import NeRFNetwork
    elif opt.tcnn:
        opt.fp16 = False
        assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"
        from nerf.network_tcnn import NeRFNetwork
    else:
        from nerf.network import NeRFNetwork

    seed_everything(opt.seed)

    model = NeRFNetwork(
        encoding="hashgrid",
        bound=opt.bound,
        cuda_ray=opt.cuda_ray,
        density_scale=1,
        min_near=opt.min_near,
        density_thresh=opt.density_thresh,
        bg_radius=opt.bg_radius,
    )

    model.eval()
    metrics = [PSNRMeter(),]
    criterion = torch.nn.MSELoss(reduction='none')
    trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)
    dataset = NeRFDataset(opt, device=device, type='test')        #Importing dataset in order to get the same camera intrinsics as training
    ### -----  END OF TORCH-NGP SPECIFIC ----- #

    ### ----- NERF-NAV PARAMETERS ----- #

    ### ESTIMATOR CONFIGS
    dil_iter = 3        # Number of times to dilate mask around features in observed image
    kernel_size = 5     # Kernel of dilation 
    batch_size = 1024   # How many rays to sample in dilated mask
    lrate_relative_pose_estimation = 1e-3       # State estimator learning rate
    N_iter = 300        # Number of times to perform gradient descent in state estimator

    #Remark: We don't have a measurement noise covariance, or rather we just set it to identity since it's not clear
    #what a covariance on a random batch of pixels should be. 
    sig0 = 1*np.eye(12)     # Initial state covariance
    Q = 1*np.eye(12)        # Process noise covariance

    ### AGENT CONFIGS

    # Extent of the agent body, centered at origin.
    # low_x, high_x
    # low_y, high_y
    # low_z, high_z
    body_lims = np.array([
        [-0.05, 0.05],
        [-0.05, 0.05],
        [-0.02, 0.02]
    ])

    # Discretizations of sample points in x,y,z direction
    body_nbins = [10, 10, 5]

    mass = 1.           # mass of drone
    g = 10.             # gravitational constant
    I = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]   # inertia tensor
    path = 'sim_img_cache/'     # Directory where pose and images are exchanged
    blend_file = 'stonehenge.blend'     # Blend file of your scene

    ### PLANNER CONFIGS
    # X, Y, Z
    #STONEHENGE
    start_pos = [0.39, -0.67, 0.2]      # Starting position [x,y,z]
    end_pos = [-0.4, 0.55, 0.16]        # Goal position
    
    # start_pos = [-0.09999999999999926,
    #             -0.8000000000010297,
    #             0.0999999999999695]
    # end_pos = [0.10000000000000231,
    #             0.4999999999996554,
    #             0.09999999999986946]

    # Rotation vector
    start_R = [0., 0., 0.0]     # Starting orientation (Euler angles)
    end_R = [0., 0., 0.0]       # Goal orientation

    # Angular and linear velocities
    init_rates = torch.zeros(3) # All rates

    T_final = 2.                # Final time of simulation
    steps = 20                  # Number of time steps to run simulation

    planner_lr = 0.001          # Learning rate when learning a plan
    epochs_init = 2500          # Num. Gradient descent steps to perform during initial plan
    fade_out_epoch = 0
    fade_out_sharpness = 10
    epochs_update = 250         # Num. grad descent steps to perform when replanning

    ### MPC CONFIGS
    mpc_noise_mean = [0., 0., 0., 0, 0, 0, 0, 0, 0, 0, 0, 0]    # Mean of process noise [positions, lin. vel, angles, ang. rates]
    mpc_noise_std = [2e-2, 2e-2, 2e-2, 1e-2, 1e-2, 1e-2, 2e-2, 2e-2, 2e-2, 1e-2, 1e-2, 1e-2]    # standard dev. of noise

    ### Integration
    start_pos = torch.tensor(start_pos).float()
    end_pos = torch.tensor(end_pos).float()

    # Change rotation vector to rotation matrix 3x3
    start_R = vec_to_rot_matrix( torch.tensor(start_R))
    end_R = vec_to_rot_matrix(torch.tensor(end_R))

    # Convert 12 dimensional to 18 dimensional vec
    start_state = torch.cat( [start_pos, init_rates, start_R.reshape(-1), init_rates], dim=0 )
    end_state   = torch.cat( [end_pos,   init_rates, end_R.reshape(-1), init_rates], dim=0 )

    #Store configs in dictionary
    planner_cfg = {
    "T_final": T_final,
    "steps": steps,
    "lr": planner_lr,
    "epochs_init": epochs_init,
    "fade_out_epoch": fade_out_epoch,
    "fade_out_sharpness": fade_out_sharpness,
    "epochs_update": epochs_update,
    'start_state': start_state.to(device),
    'end_state': end_state.to(device),
    'exp_name': opt.workspace,                  # Experiment name
    'I': torch.tensor(I).float().to(device),
    'g': g,
    'mass': mass,
    'body': body_lims,
    'nbins': body_nbins
    }

    agent_cfg = {
    'dt': T_final/steps,        # Duration of each time step
    'mass': mass,
    'g': g,
    'I': torch.tensor(I).float().to(device)
    }

    camera_cfg = {
    'half_res': False,      # Half resolution
    'white_bg': True,       # White background
    'path': path,           # Directory where pose and images are stored
    'res_x': 800,           # x resolution (BEFORE HALF RES IS APPLIED!)
    'res_y': 800,           # y resolution
    'trans': True,          # Boolean    (Transparency)
    'mode': 'RGBA'          # Can be RGB-Alpha, or just RGB
    }

    blender_cfg = {
    'blend_path': blend_file,
    'script_path': 'viz_func.py'        # Path to Blender script
    }

    filter_cfg = {
    'dil_iter': dil_iter,
    'batch_size': batch_size,
    'kernel_size': kernel_size,
    'lrate': lrate_relative_pose_estimation,
    'N_iter': N_iter,
    'sig0': torch.tensor(sig0).float().to(device),
    'Q': torch.tensor(Q).float().to(device),
    'render_viz': True,
    'show_rate': [20, 100]
    }

    extra_cfg = {
    'mpc_noise_std': torch.tensor(mpc_noise_std),
    'mpc_noise_mean': torch.tensor(mpc_noise_mean)
    }

    # Defining crucial functions related to querying the NeRF. 

    # Querying the density (for the planner)
    #In NeRF training, the camera is pointed along positive z axis, whereas Blender assumes -z, hence we need to rotate the pose
    rot = torch.tensor([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]], device=device, dtype=torch.float32)
    
    # Grabs density from NeRF Neural Network
    density_fn = lambda x: model.density(x.reshape((-1, 3)) @ rot)['sigma'].reshape(x.shape[:-1])

    # Rendering from the NeRF functions
    render_fn = lambda rays_o, rays_d: model.render(rays_o, rays_d, staged=True, bg_color=1., perturb=False, **vars(opt))
    get_rays_fn = lambda pose: get_rays(pose, dataset.intrinsics, dataset.H, dataset.W)

    # Main loop
    simulate(planner_cfg, agent_cfg, filter_cfg, extra_cfg, density_fn, render_fn, get_rays_fn)
    
    # Visualize trajectories in Blender
    bevel_depth = 0.02      # Size of the curve visualized in blender
    subprocess.run(['blender', blend_file, '-P', 'viz_data_blend.py', '--', opt.workspace, str(bevel_depth)])

    end_text = 'End of simulation'
    print(f'{end_text:.^20}')

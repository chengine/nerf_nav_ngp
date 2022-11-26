import bpy
import sys, os
from mathutils import Matrix, Vector
import numpy as np
import json
import glob
    
### PARAMS ###
if __name__ == "__main__":
    argv = sys.argv
    try:
        index = argv.index("--") + 1
    except ValueError:
        index = len(argv)

    argv = argv[index:]

    exp_name = argv[0]
    bevel_depth = argv[1]

    basefolder = bpy.path.abspath('//') + f'paths/{exp_name}'

    project = bpy.data.collections.new(f'{exp_name}_visualization')
    bpy.context.scene.collection.children.link(project)

    def add_curve(project, data, time_step=None, bevel_depth=0.02):
        # data: Nx3 (N is trajectory length)
        # make a new curve
        crv = bpy.data.curves.new('crv', 'CURVE')
        crv.dimensions = '3D'

        # make a new spline in that curve
        spline = crv.splines.new(type='NURBS')

        # a spline point for each point
        spline.points.add(len(data)-1) # theres already one point by default

        # assign the point coordinates to the spline points
        for p, new_co in zip(spline.points, data):
            p.co = (new_co.tolist() + [1.0]) # (add nurbs weight)

        # make a new object with the curve
        if time_step is None:
            obj = bpy.data.objects.new(f'traj_init', crv)
        else:
            obj = bpy.data.objects.new(f'traj_{time_step}', crv)

        obj.data.bevel_depth = bevel_depth
        project.objects.link(obj)
        bpy.context.view_layer.update()

    def poses2loc(poses):
        return poses[:, :3, -1]

    # Visualize the initial trajectory
    init_files = glob.glob(basefolder + '/init_poses/*.json')
    num_init_files = len(init_files)
    latest_init = basefolder + f'/init_poses/{num_init_files-1}.json'
    print(f'Reading {latest_init}')

    with open(latest_init, 'r') as f:
        meta = json.load(f)
    init_plan = np.array(meta["poses"])
    init_plan = poses2loc(init_plan)
    add_curve(project, init_plan, bevel_depth=0.02)

    # Visualize all subsequent plans
    time_step = 0
    while os.path.exists(basefolder + f'/replan_poses/0_time{time_step}.json'):
        replan_files = glob.glob(basefolder + f'/replan_poses/*_time{time_step}.json')
        num_replan_files = len(replan_files)
        latest_replan = basefolder + f'/replan_poses/{num_replan_files-1}_time{time_step}.json'
        print(f'Reading {latest_replan}')

        with open(latest_replan, 'r') as f:
            meta = json.load(f)
        replan_plan = np.array(meta["poses"])
        replan_plan = poses2loc(replan_plan)
        add_curve(project, replan_plan, time_step=time_step, bevel_depth=0.02)
        time_step += 1

    print("--------------------    DONE WITH BLENDER SCRIPT    --------------------")
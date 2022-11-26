import bpy
import sys
from mathutils import Matrix
import json
import numpy as np

if __name__ == "__main__":
    argv = sys.argv
    try:
        index = argv.index("--") + 1
    except ValueError:
        index = len(argv)

    argv = argv[index:]

    # argv should only be the path to a json file containing camera info
    path_to_args = argv[0]
    path_to_saves = argv[1]

    # Where to look and where to save
    arg_path = bpy.path.abspath('//') + path_to_args
    save_path = bpy.path.abspath('//') + path_to_saves

    scene = bpy.context.scene
    camera = bpy.data.objects['Camera']

    try:
        with open(arg_path,"r") as f:
            meta = json.load(f)
    except Exception as err:
        print(f"Unexpected {err}, {type(err)}")
        raise

    pose = np.array(meta['pose'])
    res_x = meta['res_x']           # x resolution
    res_y = meta['res_y']           # y resolution
    transparent = meta['trans']     # Boolean
    mode = meta['mode']             # Must be either 'RGB' or 'RGBA'

    camera.matrix_world = Matrix(pose)
    bpy.context.view_layer.update()

    # save image from camera
    bpy.context.scene.render.filepath = path_to_saves
    bpy.context.scene.render.resolution_x = res_x
    bpy.context.scene.render.resolution_y = res_y
    bpy.context.scene.render.film_transparent = transparent
    bpy.context.scene.render.image_settings.color_mode = mode
    bpy.ops.render.render(write_still = True)

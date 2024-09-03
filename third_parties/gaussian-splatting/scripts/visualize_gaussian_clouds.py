import os
from tqdm import tqdm
from plyfile import PlyData, PlyElement
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import math
from PIL import Image
import os


C0 = 0.28209479177387814
def SH2RGB(sh):
    return sh * C0 + 0.5

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

experiment_name = '7d8132ca4ef9493ba1c472d3edfb551/3dgs_experiments'
if __name__ == '__main__':
    output_path = Path('/home/yuans/home/yuans/projects/SuperGaussian/data/objaverse/rendered_scenes/chair_overfit/')
    point_cloud_paths = list(sorted((output_path / experiment_name / 'point_cloud').glob("iteration*")))
    number_of_gaussians = {}
    skip_first = True
    vis = o3d.visualization.rendering.OffscreenRenderer(1920, 1080)
    # vis.scene.set_background([0.0, 0.0, 0.0, 1.0])
    vis.scene.view.set_post_processing(False)
    parameters = o3d.io.read_pinhole_camera_parameters("scripts/objaverse_open3d_pose.json")
    vis.setup_camera(parameters.intrinsic, parameters.extrinsic)
    os.makedirs(str(output_path / experiment_name / 'point_cloud_visualization'), exist_ok=True)
    for point_cloud_path in tqdm(point_cloud_paths):
        plydata = PlyData.read(point_cloud_path / 'point_cloud.ply')
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = sigmoid(np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis])

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        colors = SH2RGB(features_dc).squeeze()
        bg = np.array([1, 1, 1]) # USE WHITE background

        arr = colors #* opacities + bg * (1 - opacities)

        # scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        # scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        # scales = np.zeros((xyz.shape[0], len(scale_names)))
        # for idx, attr_name in enumerate(scale_names):
        #     scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        #
        # rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        # rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        # rots = np.zeros((xyz.shape[0], len(rot_names)))
        # for idx, attr_name in enumerate(rot_names):
        #     rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # if not skip_first:
        #     boxes = []
        #     for scale, rot in zip(scales, rots):
        #         box = o3d.geometry.TriangleMesh.create_box(width=scale[0], height=scale[1], depth=scale[2])
        #         box = box.transform(rot)
        #         boxes.append(box)
        # else:
        #     skip_first = False
        pcd = o3d.geometry.PointCloud()
        xyz[:, 1] = -xyz[:, 1]
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(arr)

        mtl = o3d.visualization.rendering.MaterialRecord()
        mtl.shader = "defaultLit"

        vis.scene.add_geometry("pcd", pcd, mtl)
        rgb_img = np.array(vis.render_to_image())
        vis.scene.remove_geometry('pcd')
        # o3d.visualization.draw_geometries([pcd])
        iteration_index = int(point_cloud_path.name.split('_')[-1])
        number_of_gaussians[iteration_index] = len(xyz)
        Image.fromarray(rgb_img).save(str(output_path / experiment_name / 'point_cloud_visualization' / f'rendered_{iteration_index:06d}.png'))


    xs = list(sorted(number_of_gaussians.keys()))
    ys = []
    for x in xs:
        ys.append(number_of_gaussians[x])
    plt.scatter(xs, ys, s=0.4)
    plt.title(experiment_name)
    plt.xlabel('iteration')
    plt.ylabel('#gaussians')
    plt.savefig(str(output_path / experiment_name / 'number_of_gaussians.png'))
    plt.show()
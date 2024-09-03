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

experiment_name = 'chestnut_rgbd_duygu_init_0.001_fix'
if __name__ == '__main__':
    output_path = Path('output/')
    point_cloud_paths = list(sorted((output_path / experiment_name / 'point_cloud').glob("iteration*")))

    xyzs = []
    for point_cloud_path in tqdm(point_cloud_paths):
        plydata = PlyData.read(point_cloud_path / 'point_cloud.ply')
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        # opacities = sigmoid(np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis])
        #
        # features_dc = np.zeros((xyz.shape[0], 3, 1))
        # features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        # features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        # features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        # colors = SH2RGB(features_dc).squeeze()
        # bg = np.array([1, 1, 1]) # USE WHITE background
        #
        # arr = colors * opacities + bg * (1 - opacities)
        #
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
        xyzs.append(xyz)


    position_change = np.linalg.norm(np.abs(xyzs[-1] - xyzs[0]), axis=1)
    print("Position change:", np.mean(position_change), np.std(position_change), np.percentile(position_change, 10), np.percentile(position_change, 90))
    print("Done!")
    print("vertical:",  np.percentile(xyzs[0][:, 1], 95) - np.percentile(xyzs[0][:, 1], 1))

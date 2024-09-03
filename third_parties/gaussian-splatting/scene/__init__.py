#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import sys
sys.path.extend(['/home/yuan/projects/SuperGaussian/third_parties/gaussian-splatting'])

from gs_utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from gs_utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=False, resolution_scales=[1.0], initial_ply=None, use_low_res_as_gt=True, num_of_gaussians=4096, no_gt_eval=0, use_orig_gt_traj_setup=False):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.use_low_res_as_gt = use_low_res_as_gt

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        # resolution_name = 'resolution_128' if use_low_res_as_gt else 'resolution_1024'
        resolution_name = 'resolution_low' if use_low_res_as_gt else 'resolution_high'
        if os.path.exists(os.path.join(args.source_path + '/' + resolution_name, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path + '/' + resolution_name, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path + '/' + resolution_name, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path + f'/{resolution_name}', "transforms.json")):
            print("Found transforms.json file, assuming Objaverse data set!")
            scene_info = sceneLoadTypeCallbacks["Objaverse"](args.source_path + '/' + resolution_name, args.white_background,
                                                             args.eval, initial_ply, num_of_gaussians=num_of_gaussians,
                                                             no_gt_eval=no_gt_eval, use_orig_gt_traj_setup=use_orig_gt_traj_setup)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file, indent=4)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            # path = os.path.join(self.model_path,
            #                    "point_cloud",
            #                    "iteration_" + str(self.loaded_iter),
            #                    "point_cloud.ply")
            path = str(self.loaded_iter)
            self.gaussians.load_ply(path)
            print(f'loaded gaussian ply from {path}')
            # self.save(self.loaded_iter) # save initial gaussian
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
            self.save(0) # save initial gaussian

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
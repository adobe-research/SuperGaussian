import json
import os
import random
from autolab_core import CameraIntrinsics
import pickle
import numpy as np
import PIL
from plyfile import PlyData, PlyElement
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import sys
sys.path.extend(['third_parties/gaussian-splatting'])

from sg_utils.colmap_utils import qvec2rotmat
from PIL import Image

cache_path = 'cache'
def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)
    return ply_data


class MVImageNetTestDataset(Dataset):
    def __init__(
        self, version, pcd_point_size=131072, **kwargs
    ):
        super().__init__()
        self.version = version

        with open('data/mv_imagenet_category.txt', 'r') as f:
            self.category_mapping = {}
            for line in f.readlines():
                idx, category = line.strip().split(',')
                self.category_mapping[category.strip()] = int(idx.strip())

        seed = 0
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        self.testset_root = 'data/mvimgnet_testset_500'

        with open('data/supergaussian_testset.json', 'r') as f:
            self.split_scene_dirs = json.load(f)

        self.pcd_point_size = pcd_point_size
        self.transform_rgb = transforms.ToTensor()
        self.transform_depth = transforms.ToTensor()

    def __len__(self):
        return len(self.split_scene_dirs)

    def center_crop(self, data, height, width):
        h, w = data.shape[:2]
        top_left_y = (h - height) // 2
        top_left_x = (w - width) // 2
        cropped_image = data[top_left_y:top_left_y + height, top_left_x:top_left_x + width]
        return cropped_image

    def __getitem__(self, idx):
        # try:
        res = self.get_item_helper(idx)
        return res
        # except:
        #     return {'status': False, 'book_idx': -1}
    def get_item_helper(self, idx):
        scene_dir = self.split_scene_dirs[idx]
        cameras_extrinsic_file = os.path.join(self.testset_root, scene_dir, "cam_extrinsics.pkl")
        cameras_intrinsic_file = os.path.join(self.testset_root, scene_dir, "cam_intrinsics.pkl")
        xyz_file = os.path.join(self.testset_root, scene_dir, "xyz.pkl")
        rgb_file = os.path.join(self.testset_root, scene_dir, "rgb.pkl")
        with open(cameras_extrinsic_file, 'rb') as f:
            cam_extrinsics = pickle.load(f)
        with open(cameras_intrinsic_file, 'rb') as f:
            cam_intrinsics = pickle.load(f)
        with open(xyz_file, 'rb') as f:
            xyz = pickle.load(f)
        with open(rgb_file, 'rb') as f:
            rgb = pickle.load(f)

        xyz = xyz.astype(np.float32)
        ind = np.random.default_rng().choice(len(xyz), self.pcd_point_size, replace=len(xyz) < self.pcd_point_size)
        xyz = torch.from_numpy(xyz[ind]).float()
        rgb = torch.from_numpy(rgb[ind]).float()

        gt_images = []
        input_fxfycxcys = []
        cam_infos = sorted(cam_extrinsics.values(), key=lambda x: x.name)
        extrinsics = []
        input_images = []
        img_filenames = []

        for cam_info in list(cam_infos):
            input_image_path = os.path.join(self.testset_root, scene_dir, self.version, f"{cam_info.name[:-4]}.png")
            input_images.append(self.transform_rgb(Image.open(input_image_path)))
            img_filenames.append(f"{cam_info.name[:-4]}.png")
            intr = cam_intrinsics[cam_info.camera_id]
            orig_height = intr.height
            orig_width = intr.width

            image = Image.open(os.path.join(
                self.testset_root, scene_dir, "gt_rgb", f"{cam_info.name[:-4]}.png"
            ))
            image = transforms.CenterCrop(min(image.size))(image)
            gt_image = image.resize((256, 256), resample=PIL.Image.BILINEAR)
            if self.transform_rgb:
                gt_image = self.transform_rgb(gt_image)
            gt_images.append(gt_image)

            R = np.transpose(qvec2rotmat(cam_info.qvec))
            T = np.array(cam_info.tvec)
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = R.T # glm
            extrinsic[:3, 3] = T
            extrinsics.append(torch.from_numpy(extrinsic))
            intrisic = np.eye(3)
            if intr.model == 'SIMPLE_RADIAL':
                focal_length_x = intr.params[0]
                # FovY = focal2fov(focal_length_x, orig_height)
                # FovX = focal2fov(focal_length_x, orig_width)
                intrisic[0][0] = focal_length_x
                intrisic[1][1] = focal_length_x
                intrisic[0][2] = intr.params[1]
                intrisic[1][2] = intr.params[2]
            else:
                assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

            cam_intr = CameraIntrinsics("0",
                                        fx=intrisic[0][0],
                                        fy=intrisic[1][1],
                                        cx=intrisic[0][2],
                                        cy=intrisic[1][2],
                                        height=orig_height,
                                        width=orig_width)
            length = min(orig_height, orig_width)
            intrisic = cam_intr.resize(256/length)
            intrisic = intrisic.crop(height=256,
                                     width=256,
                                     crop_ci=intrisic.height//2,
                                     crop_cj=intrisic.width//2)._K
            input_fxfycxcy = [
                intrisic[0][0],
                intrisic[1][1],
                intrisic[0][2],
                intrisic[1][2],
            ]
            input_fxfycxcys.append(torch.tensor(input_fxfycxcy))
        for new_traj_i in range(2):  # the first one is for evaluation, the second one is additional trajectory used together with orig traj for vsr.
            with open(os.path.join(self.testset_root, scene_dir, 'novel_trajectory_poses', f'{new_traj_i:03d}.json'), 'r') as f:
                w2cs = json.load(f)
                w2cs = np.array(w2cs)
            for w2c_i, w2c in enumerate(w2cs):
                R = w2c[:3, :3]
                T = w2c[:3, 3]
                extrinsic = np.eye(4)
                extrinsic[:3, :3] = R
                extrinsic[:3, 3] = T
                extrinsics.append(torch.from_numpy(extrinsic))
                input_fxfycxcys.append(torch.tensor(input_fxfycxcy)) # use the last one for novel view pose
                input_images.append(self.transform_rgb(Image.open(os.path.join(self.testset_root, scene_dir,
                                                                               self.version, f'traj_{new_traj_i}_{w2c_i:03d}.png'))))

                if new_traj_i == 0:
                    gt_images.append(self.transform_rgb(Image.open(os.path.join(self.testset_root, scene_dir,
                                                                               "HR_131072_gaussian", f'traj_{new_traj_i}_{w2c_i:03d}.png')).resize((256, 256), resample=Image.BILINEAR)))
                img_filenames.append(f'traj_{new_traj_i}_{w2c_i:03d}.png')

        extrinsics = torch.stack(extrinsics).float()
        high_res_images = torch.stack(gt_images).float()
        input_images = torch.stack(input_images).float()
        input_fxfycxcys = torch.stack(input_fxfycxcys).float()
        return {"book_idx": idx, "images": input_images,
                "extrinsics": extrinsics, "fxfycxcy": input_fxfycxcys, "scene_dir": scene_dir, "xyz": xyz, "rgb": rgb,
                "orig_size": np.array([orig_height, orig_width]),
                "high_res_images": high_res_images,
                "img_filenames": img_filenames,
                'status': 1}

#
# if __name__ == "__main__":
#     objaverse_dataset = MVImageNetImageUpsamplerTestDataset(None, 4096, True,4, 0)
#     dataloader = DataLoader(
#         objaverse_dataset,
#         batch_size=1,
#         shuffle=False,
#         num_workers=1,
#         pin_memory=False,  # https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
#     )
#
#     for i, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
#         pass
#
#     print('Done')
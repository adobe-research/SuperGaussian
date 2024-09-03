from plyfile import PlyData, PlyElement
import json
from multiprocessing import Array, Value
from sg_utils.sg_helper import *
import multiprocessing
import os
import random
import torch
import time
from dataclasses import dataclass
import numpy as np
from PIL import Image
import tyro
from dataset.mvimg_test_dataset import MVImageNetTestDataset
from torch.utils.data import DataLoader

DEBUG = 0 # change to 0 for multiprocess, 1 for single process.
@dataclass
class Args:
    curr_division_id: int = 0
    division: int = 1
    novel_trajectory_num: int = 1

    cache_path: str = '/mnt/localssd'
    """Path to the scene cache"""

    video_upsampling_prior = 'realbasicvsr'

    #"/home/yuan/projects/SuperGaussian/data/demo/gaussian_demo/Nathan_Guitar_Crop.ply"
    procedure: tuple = ('video_upsampling', 'fitting_with_3dgs', ) # operations from video_upsampling, bilinear_X, fitting_with_3dgs) # operations from video_upsampling, bilinear_X, fitting_with_3dgs

    optimization_step = 2000
    """optimization steps for the gaussian splats"""

    workers_per_gpu: int = (1 if DEBUG else 1)
    """number of workers per gpu"""

    num_gpus: int = (1 if DEBUG else -1)
    """number of gpus to use. """

    gaussian_version: str = 'LR_131072_gaussian'
    """version of the gaussian splats"""

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


def prepare_rgbd(path, point_size, xyz, rgb):
    ply_path = os.path.join(path, f"surface_pcd_{point_size}_seed_0.ply")
    xyz = xyz.numpy()
    rgb = rgb.numpy()
    storePly(ply_path, xyz, rgb)
    return ply_path

def worker(
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
    num_of_gaussians: int,
    gpu_meta,
    lock
) -> None:
    while True:
        item = queue.get()
        if item is None:
            break
        # Perform some operation on the item
        batch, gpu_i = item
        curr_output_path = f"{os.path.dirname(os.path.abspath(__file__))}/test_results/{batch['scene_dir'][0]}/super_gaussian_with_realbasicvsr"

        os.makedirs(curr_output_path, exist_ok=True)
        with open(f"{curr_output_path}/config.json", 'w+') as f:
            json.dump(args.__dict__, f, indent=4)

        try:
            # dump to super_gaussian scene format (nerfstudio)
            image_folder = curr_output_path + f'/64x64'
            os.makedirs(image_folder, exist_ok=True)
            image_paths = []
            for image_i, image in enumerate(batch['images'][0]):
                image_path = image_folder + "/{}".format(batch["img_filenames"][image_i][0])
                Image.fromarray(np.clip(image.numpy() * 255, 0, 255).transpose(1,2,0).astype(np.uint8)).save(
                    image_path)
                image_paths.append(image_path)

            high_res_image_folder = curr_output_path + f'/gt/high_res_images'
            os.makedirs(high_res_image_folder, exist_ok=True)
            for image_i, image in enumerate(batch['high_res_images'][0]):
                image_path = high_res_image_folder + "/{}".format(batch["img_filenames"][image_i][0])
                Image.fromarray(np.clip(image.numpy() * 255, 0, 255).transpose(1,2,0).astype(np.uint8)).save(
                    image_path)

            res = {
                "camera_model": "OPENCV",
                "fl_x": batch['fxfycxcy'][0, 0, 0].item(),
                "fl_y": batch['fxfycxcy'][0, 0, 1].item(),
                "cx": batch['fxfycxcy'][0, 0, 2].item(),
                "cy": batch['fxfycxcy'][0, 0, 3].item(),
                "w": 256,
                "h": 256,
                "k1": 0.0,
                "k2": 0.0,
                "k3": 0.0,
                "k4": 0.0,
                "p1": 0.0,
                "p2": 0.0,
                "frames": []
            }

            for image_i, path in enumerate(image_paths):
                c2w = torch.linalg.inv(batch["extrinsics"][0][image_i]).numpy()
                c2w[:3, 1:3] *= -1
                res['frames'].append({
                    "file_path": '/'.join(path.replace('64x64', 'images').split('/')[-2:]),
                    "transform_matrix": c2w.tolist(),
                })
            transform_path = f"{curr_output_path}/transforms.json"
            with open(transform_path, 'w+') as f:
                json.dump(res, f, indent=4)
            print("step 0 completes!")
        except Exception as e:
            print(str(e))
            return
        \
        initial_pcd_path = prepare_rgbd(curr_output_path, num_of_gaussians, batch['xyz'][0], batch['rgb'][0])

        step = 1
        latest_resolution = batch['images'].shape[4] # assume it's square image
        latest_res_path = image_folder
        latest_gaussian_ckpt = None
        for plan in args.procedure:
            curr_output_path = f"{os.path.dirname(os.path.abspath(__file__))}/test_results/{batch['scene_dir'][0]}/super_gaussian_with_realbasicvsr/step_{step}_{plan}"
            if plan == 'video_upsampling':
                output_resolution = latest_resolution * 4 # no matter what upsampling scale of the prior, it will force to output this resolution
                run_video_upsampling(args.video_upsampling_prior, gpu_i,
                                     latest_res_path,
                                     f"{curr_output_path}/{output_resolution}x{output_resolution}",
                                     output_resolution,
                                     "single_video"
                                     )
                latest_resolution = output_resolution
                latest_res_path = f"{curr_output_path}/{output_resolution}x{output_resolution}"
            elif 'bilinear' in plan:
                output_resolution = int(plan.split('_')[-1])
                run_bilinear_resampling(output_resolution,
                                        latest_res_path,
                                        f"{curr_output_path}/{output_resolution}x{output_resolution}")
                latest_resolution = output_resolution
                latest_res_path = f"{curr_output_path}/{output_resolution}x{output_resolution}"
            elif plan == 'fitting_with_3dgs':
                # assert latest_resolution == 2048
                optimization_step = args.optimization_step if latest_gaussian_ckpt is None else args.restoration_optimization_step
                fitting_with_3dgs(latest_res_path,
                                  high_res_image_folder,
                                  transform_path,
                                  initial_pcd_path,
                                  latest_gaussian_ckpt,
                                  optimization_step,
                                  curr_output_path,
                                  num_of_gaussians,
                                  gpu_i)
                latest_res_path = f"{curr_output_path}/point_cloud/iteration_{optimization_step}/predicted"
                latest_gaussian_ckpt = f"{curr_output_path}/point_cloud/iteration_{optimization_step}/point_cloud.ply"
            else:
                raise NotImplementedError
            print(f"step {step} ({plan}) completes!")
            step += 1
        with lock.get_lock():
            gpu_meta[gpu_i] = gpu_meta[gpu_i] - 1

        with count.get_lock():
            count.value += 1

        queue.task_done()


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = tyro.cli(Args)
    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)
    mv_imagenet_dataset = MVImageNetTestDataset(args.gaussian_version)
    # Start worker processes on each of the GPUs
    if args.num_gpus == -1:
        args.num_gpus = torch.cuda.device_count()
    args.num_of_gaussians = int(args.gaussian_version.split('_')[1])
    gpu_meta = Array('i', [0] * args.num_gpus)
    lock = Value('i', 0)

    for gpu_i in range(args.num_gpus):
        for worker_i in range(args.workers_per_gpu):
            process = multiprocessing.Process(
                target=worker, args=(queue, count, args.num_of_gaussians, gpu_meta, lock
                                     )
            )
            process.daemon = True
            process.start()

    # Add items to the queue
    random.seed(0)
    np.random.seed(0)
    dataloader = DataLoader(
        mv_imagenet_dataset,
        batch_size=1, # DO NOT CHANGE
        shuffle=False,
        num_workers=(0 if DEBUG else 4),
        # prefetch_factor=(1 if DEBUG else 32),
        # persistent_workers=True,
        # pin_memory=False,  # https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
    )

    for idx, batch in enumerate(dataloader):
        gpu = idx % args.num_gpus
        while True:
            with lock.get_lock():
                if gpu_meta[gpu] < args.workers_per_gpu:
                    queue.put((batch, gpu))
                    gpu_meta[gpu] = 1 + gpu_meta[gpu]
                    break
            gpu = (gpu + 1) % args.num_gpus
            time.sleep(0.1)

        print(f"Finished {count.value}/{len(mv_imagenet_dataset)}")

    while True:
        time.sleep(5)
        print(f"Finished {count.value}/{len(mv_imagenet_dataset)}")
        if count.value == len(mv_imagenet_dataset):
            break

    # Wait for all tasks to be completed
    queue.join()
    count.value = 0
    # Add sentinels to the queue to stop the worker processes
    for i in range(args.num_gpus * args.workers_per_gpu):
        queue.put(None)




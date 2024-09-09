import shutil
import subprocess
from pathlib import Path
from PIL import Image
import os
import numpy as np

def run_bilinear_resampling(target_size, input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    for img_path in sorted(Path(input_path).glob("*.png")):
        Image.open(str(img_path)).resize(size=(target_size, target_size), resample=Image.BILINEAR).save(os.path.join(output_path, img_path.name))
    print(f"run bilinear upsampling to resample video to ({target_size},{target_size}) from {input_path} to {output_path}")


def run_video_upsampling(video_upsampling_prior, gpu_i, input_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    if video_upsampling_prior == 'realbasicvsr':
        command = (
            f"cd third_parties/RealBasicVSR && " 
            f"CUDA_VISIBLE_DEVICES={gpu_i} /root/miniconda3/envs/realbasicvsr/bin/python inference_realbasicvsr_4x.py " # change to realbasicvsr conda env python path
            f"configs/realbasicvsr_x4.py "
            f"checkpoints/RealBasicVSR_x4.pth {input_path} {output_path} "
        )
        print(command)
        ret = subprocess.run(command, shell=True, executable='/bin/bash', stdout=subprocess.DEVNULL, check=True)

    else:
        raise NotImplementedError
    print(f"run video upsampling with prior {video_upsampling_prior} from {input_path} to {output_path}")


def fitting_with_3dgs(image_path, gt_img_path, transform_path, initial_pcd_path, latest_gaussian_ckpt, optimization_step, output_path, num_of_gaussians, gpu_i):
    # creat soft link
    os.makedirs(os.path.join(output_path, 'resolution_low', 'gt'), exist_ok=True)
    if not os.path.exists(output_path + '/resolution_low/transforms.json'):
        os.symlink(transform_path, output_path + '/resolution_low/transforms.json')
    if not os.path.exists(output_path + '/resolution_low/images'):
        os.symlink(image_path, output_path + '/resolution_low/images')
    if not os.path.exists(output_path + '/resolution_low/gt/high_res_images') and gt_img_path is not None:
        os.symlink(gt_img_path, output_path + '/resolution_low/gt/high_res_images')
    fn = initial_pcd_path.split('/')[-1]
    if not os.path.exists(output_path + f'/resolution_low/{fn}'):
        os.symlink(initial_pcd_path, output_path + f'/resolution_low/{fn}')

    command = (
        f"cd third_parties/gaussian-splatting && "
        f"CUDA_VISIBLE_DEVICES={gpu_i} /root/miniconda3/envs/super_gaussian_eccv24/bin/python train.py " # change to your python environment
        f"--exp_name {output_path} "
        f"--iterations {optimization_step} "
        f"-s {output_path} --use_low_res_as_gt --num_of_gaussians {num_of_gaussians} -r 1"
    )
    print(command)
    ret = subprocess.run(command, shell=True, executable='/bin/bash', check=True) # stdout=subprocess.DEVNULL,
    print(f"run fitting with 3dgs")


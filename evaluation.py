import json
import os
import pickle
import torchvision.transforms as transforms
import PIL
import multiprocessing
import pyiqa
from PIL import Image
import numpy as np
import torch
from pathlib import Path
import time
from tqdm import tqdm
from torchmetrics.image.inception import InceptionScore

def worker(
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
) -> None:
    while True:
        scene_dir  = queue.get()
        if scene_dir == 'Done':
            break
        get_images(scene_dir)
        with count.get_lock():
            count.value += 1

        queue.task_done()


def get_images(scene_dir):
    print(dataset_root, scene_dir)
    cameras_extrinsic_file = os.path.join(dataset_root, "mvimgnet_testset_500", scene_dir, "cam_extrinsics.pkl")
    with open(cameras_extrinsic_file, 'rb') as f:
        cam_extrinsics = pickle.load(f)
    cam_infos = sorted(cam_extrinsics.values(), key=lambda x: x.name)
    for cam_info in list(cam_infos):
        filename = f"traj_0_{(int(cam_info.name[:-4])-1):03d}.png"
        image_path = os.path.join(
            dataset_root, "mvimgnet_testset_500", scene_dir, "HR_131072_gaussian", filename
        )

        image = Image.open(image_path)
        # image = transforms.CenterCrop(min(image.size))(image)
        gt_image = image.resize(
            (256, 256), resample=PIL.Image.BILINEAR
        )
        name = scene_dir.replace('/', '_') + '_' + cam_info.name[:-4] + '.png'
        try:
            if os.path.exists(os.path.join(gt_cache_path, name)):
                Image.open(os.path.join(gt_cache_path, name))
            else:
                gt_image.save(os.path.join(gt_cache_path, name))
        except:
            gt_image.save(os.path.join(gt_cache_path, name))

        # load pred
        # filename = f"{(int(cam_info.name[:-4]) - 1):03d}.png"
        image_path = os.path.join(
            target_root, scene_dir, f"super_gaussian_with_{target}/step_2_fitting_with_3dgs/point_cloud/iteration_2000/predicted", filename
        )
        image = Image.open(image_path)

        try:
            if os.path.exists(os.path.join(pred_cache_path, name)):
                Image.open(os.path.join(pred_cache_path, name))
            else:
                image.save(os.path.join(pred_cache_path, name))
        except:
            image.save(os.path.join(pred_cache_path, name))

        image_path = os.path.join(
            dataset_root, "mvimgnet_testset_500", scene_dir, "gt_rgb", cam_info.name[:-4] + '.png'
        )

        image = Image.open(image_path)

        image = transforms.CenterCrop(min(image.size))(image)
        gt_image = image.resize(
            (256, 256), resample=PIL.Image.BILINEAR
        )
        name = scene_dir.replace('/', '_') + '_' + cam_info.name[:-4] + '.png'

        try:
            if os.path.exists(os.path.join(fid_gt_cache_path, name)):
                Image.open(os.path.join(fid_gt_cache_path, name))
            else:
                gt_image.save(os.path.join(fid_gt_cache_path, name))
        except:
            gt_image.save(os.path.join(fid_gt_cache_path, name))


if __name__ == '__main__':
    target = 'realbasicvsr' # 'realbasicvsr' or 'gigagan' or 'videogigagan'
    num_workers = 16

    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)
    dataset_root = 'data'
    target_root = f'{target}_test_results'
    fid_gt_cache_path = f'evaluations/{target}_prior/gt_fid'
    gt_cache_path = f'evaluations/{target}_prior/gt'
    pred_cache_path = f'evaluations/{target}_prior/pred'
    os.makedirs(gt_cache_path,exist_ok=True)
    os.makedirs(pred_cache_path,exist_ok=True)
    os.makedirs(fid_gt_cache_path,exist_ok=True)

    with open(f'{dataset_root}/mv_imagenet_category.txt', 'r') as f:
        category_mapping = {}
        for line in f.readlines():
            idx, category = line.strip().split(',')
            category_mapping[category.strip()] = int(idx.strip())

    split_scene_dirs = []
    categories = [str(elem) for elem in list(sorted(category_mapping.values()))]


    with open(f'{dataset_root}/supergaussian_testset.json', 'r') as f:
        split_scene_dirs = json.load(f)

    for worker_i in range(num_workers):
        process = multiprocessing.Process(
            target=worker, args=(queue, count,)
        )
        process.daemon = True
        process.start()

    for scene_dir in tqdm(split_scene_dirs):
        queue.put((scene_dir))

    while True:
        time.sleep(5)
        print(f"Finished {count.value}/{len(split_scene_dirs)}")
        if count.value == len(split_scene_dirs):
            break

    # Wait for all tasks to be completed
    queue.join()
    count.value = 0
    # Add sentinels to the queue to stop the worker processes
    for i in range(num_workers):
        queue.put('Done')


    fid_metric = pyiqa.create_metric('lpips')

    scores = []
    for fn in tqdm(Path(gt_cache_path).glob('*.png'), total=len(list(Path(gt_cache_path).glob('*.png')))):
        score = fid_metric(str(fn), str(fn).replace('gt', 'pred'))
        scores.append(score)
    print(f'{target} LPIPS: {torch.stack(scores).mean()}')


    fid_metric = pyiqa.create_metric('niqe')

    scores = []
    for fn in tqdm(Path(pred_cache_path).glob('*.png'), total=len(list(Path(pred_cache_path).glob('*.png')))):
        score = fid_metric(str(fn))
        scores.append(score)
    print(f'{target} NIQE: {torch.stack(scores).mean()}')


    inception = InceptionScore()
    img_list = list(sorted(Path(pred_cache_path).glob('*.png')))
    import random
    random.seed(0)
    fid_metric = pyiqa.create_metric('fid')
    print("FID: ", fid_metric(fid_gt_cache_path, pred_cache_path))
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
import json
import os
import cv2
import torch
import pyiqa
from random import randint
import torch.nn.functional as F
from gs_utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from gs_utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from PIL import Image
from torchvision import transforms
import numpy as np


def training(dataset, opt, pipe, checkpoint, debug_from, load_iteration, initial_ply, use_low_res_as_gt,
             num_of_gaussians, no_gt_eval=0, gaussian_regularization=None, dump_img=True,
             use_orig_gt_traj_setup=False, upsample_every_n_step=50):
    first_iter = 0
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    lpips_metric = pyiqa.create_metric('lpips', device=device)
    psnr_metric = pyiqa.create_metric('psnr', device=device)
    niqe_metric = pyiqa.create_metric('niqe', device=device)
    ssim_metric = pyiqa.create_metric('ssim', device=device)

    prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, gaussian_regularization=gaussian_regularization)
    scene = Scene(dataset, gaussians, load_iteration=load_iteration, initial_ply=initial_ply,
                  use_low_res_as_gt=use_low_res_as_gt, num_of_gaussians=num_of_gaussians,
                  no_gt_eval=no_gt_eval, use_orig_gt_traj_setup=use_orig_gt_traj_setup)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print(f'Loaded checkpoint from {checkpoint}')

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.warmup_iterations + opt.iterations), desc="Training progress")
    first_iter += 1

    # Log and save
    # training_report(0, 0, 0, l1_loss, 0, scene, render, (pipe, background),
    #                 dump_img=True,
    #                 # iteration == opt.iterations,
    #                 niqe_metric=niqe_metric,
    #                 lpips_metric=lpips_metric,
    #                 psnr_metric=psnr_metric,
    #                 ssim_metric=ssim_metric,
    #                 use_orig_gt_traj_setup=use_orig_gt_traj_setup)

    for iteration in range(first_iter, opt.warmup_iterations + opt.iterations + 1):

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations+opt.warmup_iterations:
                progress_bar.close()

            # Log and save
            training_report(iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), scene, render, (pipe, background), dump_img=(iteration == opt.iterations+opt.warmup_iterations) and dump_img, #iteration == opt.iterations,
                            niqe_metric=niqe_metric,
                            lpips_metric=lpips_metric,
                            psnr_metric=psnr_metric,
                            ssim_metric=ssim_metric,
                            use_orig_gt_traj_setup=use_orig_gt_traj_setup)


            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations + opt.warmup_iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # # Create Tensorboard writer
    # tb_writer = None
    # if TENSORBOARD_FOUND:
    #     tb_writer = SummaryWriter(args.model_path)
    # else:
    #     print("Tensorboard not available: not logging progress")
    # return tb_writer

def training_report(iteration, Ll1, loss, l1_loss, elapsed, scene : Scene, renderFunc, renderArgs, dump_img=False,
                    niqe_metric=None, lpips_metric=None, psnr_metric=None, ssim_metric=None, use_orig_gt_traj_setup=False):
    # if tb_writer:
    #     tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
    #     tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
    #     tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    with torch.no_grad():
        # if iteration % 500 == 0:
        if dump_img:
            torch.cuda.empty_cache()
            if use_orig_gt_traj_setup:
                validation_configs = ({'name': 'train', 'cameras': scene.getTrainCameras()},)
            else:
                validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},)

            for config in validation_configs:
                if config['cameras'] and len(config['cameras']) > 0:
                    if dump_img:
                        predicted_img_path = os.path.join(scene.model_path, "point_cloud", f"iteration_{iteration}/predicted")
                        # gt_img_path = os.path.join(scene.model_path, "point_cloud", f"iteration_{iteration}/pseudo_gt")
                        os.makedirs(predicted_img_path, exist_ok=True)
                        # os.makedirs(gt_img_path, exist_ok=True)

                    for mode in ['novel']:
                        viewpoints = []
                        preds = []
                        gts = []
                        l1_test = 0.0
                        if mode == 'novel':
                            for camera in config['cameras']:
                                if 'traj_0' in camera.image_name or use_orig_gt_traj_setup:
                                    viewpoints.append(camera)
                        else:
                            raise NotImplementedError
                        if len(viewpoints) == 0:
                            continue
                        for idx, viewpoint in enumerate(viewpoints):
                            image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                            if hasattr(viewpoint, 'gt_high_res_image'):
                                gt_image = torch.clamp(viewpoint.gt_high_res_image.to("cuda"), 0.0, 1.0)
                            else:
                                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                            # if tb_writer and (idx < 5):
                            #     tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            #     if iteration == 0:
                            #         tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if dump_img:
                                Image.fromarray(np.clip(image.detach().cpu().permute(1,2,0).numpy() * 255., 0, 255).astype(np.uint8)).save(predicted_img_path + f'/{viewpoint.image_name}.png')
                                # Image.fromarray(np.clip(gt_image.detach().cpu().permute(1,2,0).numpy() * 255., 0, 255).astype(np.uint8)).save(gt_img_path + f'/{viewpoint.image_name}.png')
                            preds.append(image.detach())
                            gts.append(gt_image.detach())
                            l1_test += l1_loss(image, gt_image).mean().float()
                        preds = torch.stack(preds)
                        gts = torch.stack(gts)
                        niqe_test = niqe_metric(preds, gts).mean().float()
                        lpips_test = lpips_metric(preds, gts).mean().float()
                        psnr_test = psnr_metric(preds, gts).mean().float()
                        ssim_test = ssim_metric(preds, gts).mean().float()
                        l1_test /= len(config['cameras'])
                        print("\n[ITER {}] Evaluating {}: L1 {} PSNR {} LPIPS {} NIQE {} SSIM {}".format(iteration, config['name'], l1_test, psnr_test, lpips_test, niqe_test, ssim_test))
                        os.makedirs(os.path.join(scene.model_path, "point_cloud", f"iteration_{iteration}"), exist_ok=True)
                        with open(os.path.join(scene.model_path, "point_cloud", f"iteration_{iteration}", f"performance_{mode}.json"), 'w+') as file:
                            json.dump({'psnr': psnr_test.cpu().item(),
                                            'l1': l1_test.cpu().item(),
                                            'niqe': niqe_test.cpu().item(),
                                            'ssim': ssim_test.cpu().item(),
                                            'lpips': lpips_test.cpu().item()}, file)
                        # if tb_writer:
                        #     tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                        #     tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

            print("\n[ITER {}] Saving Gaussians".format(iteration))
            scene.save(iteration)

            # if tb_writer:
            #     # tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            #     tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
            torch.cuda.empty_cache()
        if dump_img:
            validation_configs = ({'name': 'train', 'cameras': scene.getTrainCameras()},)

            for config in validation_configs:
                if config['cameras'] and len(config['cameras']) > 0:
                    if dump_img:
                        predicted_img_path = os.path.join(scene.model_path, "point_cloud", f"iteration_{iteration}/predicted")
                        # gt_img_path = os.path.join(scene.model_path, "point_cloud", f"iteration_{iteration}/pseudo_gt")
                        os.makedirs(predicted_img_path, exist_ok=True)
                        # os.makedirs(gt_img_path, exist_ok=True)

                        preds = []
                        gts = []
                        l1_test = 0.0
                        viewpoints = config['cameras']
                        for idx, viewpoint in enumerate(viewpoints):
                            image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                            if hasattr(viewpoint, 'gt_high_res_image'):
                                gt_image = torch.clamp(viewpoint.gt_high_res_image.to("cuda"), 0.0, 1.0)
                            else:
                                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                            # if tb_writer and (idx < 5):
                            #     tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                            #     if iteration == 0:
                            #         tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if dump_img:
                                Image.fromarray(np.clip(image.detach().cpu().permute(1,2,0).numpy() * 255., 0, 255).astype(np.uint8)).save(predicted_img_path + f'/{viewpoint.image_name}.png')
                                # Image.fromarray(np.clip(gt_image.detach().cpu().permute(1,2,0).numpy() * 255., 0, 255).astype(np.uint8)).save(gt_img_path + f'/{viewpoint.image_name}.png')
                            preds.append(image.detach())
                            gts.append(gt_image.detach())
                            l1_test += l1_loss(image, gt_image).mean().float()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    # parser.add_argument('--ip', type=str, default="127.0.0.1")
    # parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    # parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    # parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    # parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gaussian_regularization", type=str, default = None)
    parser.add_argument("--exp_name", type=str, default = "")
    parser.add_argument("--load_iteration", type=str, default = None)
    parser.add_argument("--initial_ply", type=str, default = None)
    parser.add_argument("--use_low_res_as_gt", action="store_true")
    parser.add_argument("--no_gt_eval", type=int, default = 0)
    parser.add_argument("--num_of_gaussians", type=int, default = 4096)
    parser.add_argument("--use_ing2g_setup", action="store_true")
    parser.add_argument("--for_video", action="store_true")
    parser.add_argument("--for_finetuned", action="store_true")
    parser.add_argument("--use_basicvsrpp", action="store_true")
    parser.add_argument("--use_orig_gt_traj_setup", action="store_true")
    parser.add_argument("--upsample_every_n_step", type=int, default = 50)

    args = parser.parse_args(sys.argv[1:])
    # args.save_iterations.append(args.iterations)
    # args.model_path += args.exp_name
    args.model_path = args.exp_name
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    # training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.load_iteration)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.start_checkpoint, args.debug_from, args.load_iteration,
             args.initial_ply, args.use_low_res_as_gt, args.num_of_gaussians, args.no_gt_eval, args.gaussian_regularization,
             use_orig_gt_traj_setup=args.use_orig_gt_traj_setup, upsample_every_n_step=args.upsample_every_n_step)

    # All done
    print("\nTraining complete.")

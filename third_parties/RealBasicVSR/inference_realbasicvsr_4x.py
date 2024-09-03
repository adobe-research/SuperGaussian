import argparse
import glob
import os
from ast import literal_eval as make_tuple
import cv2
import mmcv
import numpy as np
import pickle
import torch
from mmcv.runner import load_checkpoint
from mmedit.core import tensor2img
from PIL import Image
from realbasicvsr.models.builder import build_model

VIDEO_EXTENSIONS = ('.mp4', '.mov')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inference script of RealBasicVSR')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('input_dir', help='directory of the input video')
    parser.add_argument('output_dir', help='directory of the input video')

    parser.add_argument(
        '--max_seq_len',
        type=int,
        default=None,
        help='maximum sequence length to be processed')

    parser.add_argument(
        '--is_save_as_png',
        type=bool,
        default=True,
        help='whether to save as png')
    parser.add_argument(
        '--fps', type=float, default=25, help='FPS of the output video')
    args = parser.parse_args()
    return args


def init_model(config, checkpoint=None):
    """Initialize a model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Which device the model will deploy. Default: 'cuda:0'.

    Returns:
        nn.Module: The constructed model.
    """

    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    config.test_cfg.metrics = None
    model = build_model(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)

    model.cfg = config  # save the config in the model for convenience
    model.eval()

    return model


def main():
    args = parse_args()
    with open('video_prior_config.pkl', 'wb') as f:  # open a text file
        pickle.dump(args, f)  # serialize the list
    # initialize the model
    model = init_model(args.config, args.checkpoint)
    for iter in range(1):  # 4x upsampling
        input_paths = sorted(glob.glob(f'{args.input_dir}/*'))
        input_maps = {}
        for input_path in input_paths:
            fn = os.path.basename(input_path)
            if 'traj' not in fn:
                if 'orig' not in input_maps:
                    input_maps['orig'] = []
                input_maps['orig'].append(input_path)
            else:
                traj_name = '_'.join(fn.split('_')[:2])
                if traj_name not in input_maps:
                    input_maps[traj_name] = []
                input_maps[traj_name].append(input_path)

        for traj_name, traj_input_paths in input_maps.items():
            inputs = []

            for input_path in traj_input_paths:
                img = Image.open(input_path)
                # img = mmcv.imread(input_path, channel_order='rgb')
                img = np.array(img)
                inputs.append(img)

            for i, img in enumerate(inputs):
                img = torch.from_numpy(img / 255.).permute(2, 0, 1).float()
                inputs[i] = img.unsqueeze(0)
            inputs = torch.stack(inputs, dim=1)

            # map to cuda, if available
            cuda_flag = False
            if torch.cuda.is_available():
                model = model.cuda()
                cuda_flag = True

            with torch.no_grad():
                if isinstance(args.max_seq_len, int):
                    outputs = []
                    for i in range(0, inputs.size(1), args.max_seq_len):
                        imgs = inputs[:, i:i + args.max_seq_len, :, :, :]
                        if cuda_flag:
                            imgs = imgs.cuda()
                        outputs.append(model(imgs, test_mode=True)['output'].cpu())
                    outputs = torch.cat(outputs, dim=1)
                else:
                    if cuda_flag:
                        inputs = inputs.cuda()
                    outputs = model(inputs, test_mode=True)['output'].cpu()

            if os.path.splitext(args.input_dir)[1] in VIDEO_EXTENSIONS:
                output_dir = os.path.dirname(args.output_dir)
                mmcv.mkdir_or_exist(output_dir)

                h, w = outputs.shape[-2:]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(args.output_dir, fourcc, args.fps,
                                               (w, h))
                for i in range(0, outputs.size(1)):
                    img = tensor2img(outputs[:, i, :, :, :])
                    video_writer.write(img.astype(np.uint8))
                cv2.destroyAllWindows()
                video_writer.release()
            else:
                mmcv.mkdir_or_exist(args.output_dir)
                for i in range(0, outputs.size(1)):
                    output = tensor2img(outputs[:, i, :, :, :])
                    filename = os.path.basename(traj_input_paths[i])
                    if args.is_save_as_png:
                        file_extension = os.path.splitext(filename)[1]
                        # filename = filename.replace(file_extension, '.png')
                    mmcv.imwrite(output, f'{args.output_dir}/{filename}')
                    (Image.open(f'{args.output_dir}/{filename}')
                     .save(f'{args.output_dir}/{filename}'))


if __name__ == '__main__':
    main()

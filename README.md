# SuperGaussian: Repurposing Video Models for 3D Super Resolution

Yuan Shen, Duygu Ceylan, Paul Guerrero, Zexiang Xu, Niloy J. Mitra, Shenlong Wang, and Anna Frühstück

ECCV 2024

[Paper](https://arxiv.org/abs/2406.00609) │ [Project Page](https://supergaussian.github.io/)

TLDR: Instead of image prior, we use video upsampling prior to achieve 3D upsampling on generic low-res 3D representations. 

![Teaser](assets/teaser.gif)

In this codebase, we provide a SuperGaussian implementation on a third-parties video upsampling prior, [RealBasicVSR](https://github.com/ckkelvinchan/RealBasicVSR). 
Additionally, our codebase provides 3D upsampling evaluations on the MVImgNet testset given the upsampled image sequences from the image prior GigaGAN (image) and video prior Videogigagan (video).
Hopefully, this could help the community to reproduce our results and compare with other methods.

### Citation
Please cite our paper if you find this repo useful!
```bibtex
@inproceedings{Shen2024SuperGaussian,
  title = {SuperGaussian: Repurposing Video Models for 3D Super Resolution},
  author = {Shen, Yuan and Ceylan, Duygu and Guerrero, Paul and Xu, Zexiang and Mitra, {Niloy J.} and Wang, Shenlong and Fr{\"u}hst{\"u}ck, Anna},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year = {2024},
}
```

### Dependencies

#### 0. Hardware Requirements
- NVIDIA GPU with CUDA support 11.8. 
- The code has been tested with NVIDIA A6000 on Ubuntu 20.04.
#### 1. Install with Conda
```bash
conda create -n super_gaussian_eccv24 python=3.8 -y

conda activate super_gaussian_eccv24
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

conda install cuda -c nvidia/label/cuda-11.8.0 -y
pip install tqdm plyfile==0.8.1 hydra-core==1.3.1 h5py==3.8.0 autolab_core==1.1.1 pyiqa timm==0.9.10 rich wandb lpips boto3 tyro
export CUDA_HOME=$(dirname $(dirname $(which python)))
cd third_parties/gaussian-splatting/submodules/diff-gaussian-rasterization
rm -rf build && pip install -e .
cd ../simple-knn
rm -rf build && pip install -e .

# install RealBasicvsr in another conda env
conda create -n realbasicvsr python=3.8 -y
conda activate realbasicvsr
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html # it needs to be torch 1.7.1. For CuDA, it depends on your GPU compatibility.
pip install mmcv-full==1.5.0
pip install mmedit==0.15.0

# install Evaluation dependencies in a third conda env
conda create -n supergaussian_evaluation python=3.10 -y
conda activate supergaussian_evaluation
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install pyiqa torchmetrics
pip install torch-fidelity

conda activate super_gaussian_eccv24

# change the python environment path in sg_utils/sg_helpers.py:20 and sg_utils/sg_helpers.py:47 to your conda environment path
```

#### 2. Install with Docker (Recommended)
```bash
export DATA_PATH=[your data path (see testdata downloading section)]
docker pull yshen47/adobe_supergaussian:latest
docker run -it -v $DATA_PATH:/mnt/data --shm-size=64g --gpus all yshen47/adobe_supergaussian:latest bash
cd /root/SuperGaussian # By default, you should be at this directory
```

### Test data Download

1. Download the [MVImgNet test set](https://uofi.box.com/s/d13gifxwz573cr6li37r1m39cczrvzhm) with 523 scenes. Unzip the dataset in the project root, i.e., SuperGaussian/data. This should be the DATA_PATH to mount in your docker container. 
Each scene are grouped first by its category id and then scene id, defined by MVImgNet.
The test scenes are selected for its rich details, diversity and challenging scenarios. Within each scene folder, the directory are organized as follows. Please use our provided dataloader to load data (dataset/mvimg_test_dataset.py)
```     
--- [category_id]/[scene_id]
    --- cam_extrinsics.pkl          # camera extrinsics
    --- cam_intrinsics.pkl          # camera intrinsics
    --- rgb.pkl                     # low-res 3DGS rgb
    --- xyz.pkl                     # low-res 3DGS xyz(position)
    --- novel_trajectory_poses      # we prepare several novel trajectory poses for evaluation to avoid overfit to gt trajectory. Trajectory specified in 000.json is used for test evaluation. 
        --- 0000.json
        --- 0001.json
        ...
    --- gt_rgb                      # The original ground_truth RGB images in MVImgNet
        --- 0000.png
        --- 0001.png
        ...
    --- LR_131072_gaussian          # The low-res 3DGS renderings we prepaed in MVImgNet 
        --- 0000.png                # images without prefix are original RGB sequences in MVImgNet
        --- 0001.png
        ...
        --- traj_0_000.png          # images with prefix 'traj_0' are the novel trajectory images in MVImgNet. traj_0 is used for test evaluation.
        --- traj_0_001.png
        ...
        --- traj_1_000.png          # images with prefix 'traj_1' are the novel trajectory images in MVImgNet. traj_1 is added into the upsampling and 3D lifting.
        --- traj_1_001.png
        ...
    --- HR_131072_gaussian          # The high-res 3DGS renderings we prepared in MVImgNet ( so it can be used as 'gt' on the test trajectory)
        --- 0000.png                # images without prefix are original RGB sequences in MVImgNet
        --- 0001.png
        ...
        --- traj_0_000.png          # images with prefix 'traj_0' are the novel trajectory images in MVImgNet. traj_0 is used for test evaluation.
        --- traj_0_001.png
        ...
        --- traj_1_000.png          # images with prefix 'traj_1' are the novel trajectory images in MVImgNet. traj_1 is added into the upsampling and 3D lifting.
        --- traj_1_001.png
        ...
        
    --- gigagan_image               # The upsampled RGB images using image upsampling prior, [GigaGAN](https://mingukkang.github.io/GigaGAN/)
        --- 0000.png                # images without prefix are original RGB sequences in MVImgNet
        --- 0001.png
        ...
        --- traj_0_000.png          # images with prefix 'traj_0' are the novel trajectory images in MVImgNet. traj_0 is used for test evaluation.
        --- traj_0_001.png
        ...
        --- traj_1_000.png          # images with prefix 'traj_1' are the novel trajectory images in MVImgNet. traj_1 is added into the upsampling and 3D lifting.
        --- traj_1_001.png
        ...
    ... upsampled_rgb_videogigagan  # The upsampled RGB images using video upsampling prior, [VideoGigaGAN](https://videogigagan.github.io/)
        --- 0000.png                # images without prefix are original RGB sequences in MVImgNet
        --- 0001.png
        ...
        --- traj_0_000.png          # images with prefix 'traj_0' are the novel trajectory images in MVImgNet. traj_0 is used for test evaluation.
        --- traj_0_001.png
        ...
        --- traj_1_000.png          # images with prefix 'traj_1' are the novel trajectory images in MVImgNet. traj_1 is added into the upsampling and 3D lifting.
        --- traj_1_001.png
        ...
```
### RealBasicVSR Checkpoints Download
Download the pre-trained weights to `third_parties/RealBasicVSR/checkpoints/`. ([Dropbox](https://www.dropbox.com/s/eufigxmmkv5woop/RealBasicVSR.pth?dl=0) / [Google Drive](https://drive.google.com/file/d/1OYR1J2GXE90Zu2gVU5xc0t0P_UmKH7ID/view) / [OneDrive](https://entuedu-my.sharepoint.com/:u:/g/personal/chan0899_e_ntu_edu_sg/EfMvf8H6Y45JiY0xsK4Wy-EB0kiGmuUbqKf0qsdoFU3Y-A?e=9p8ITR))

### Evaluation
1. To run inference on all our test scenes, you can use the following command. If you do not feel like running the inference, go to 2.
```bash
conda activate super_gaussian_eccv24
python main_super_gaussian.py    # change upsampling_prior variable in Line 23 to switch between different priors.
conda activate supergaussian_evaluation
python evaluation.py             # change target variable in Line 94 to switch between different priors.
```
Note main_super_gaussian.py will create a folder in the root directory to store the upsampled 3D represented in 3DGS and final renderings for each scene. It also outputs a performance json averaged across each scene. And evaluation.py calculates performance by averaging image pairs across entire test set (instead of at scene-levels) 

2. We provide the above evaluation results for all priors, if you hope to make an Apple-to-Apple comparison between our SuperGaussian using VideoGigaGAN with your method. 
You can download from links in the table below to get all our inference results on GigaGAN, VideoGigaGAN and RealBasicVSR. You are able to access 3D upsampled gaussians 
and final 4x renderings with all poses in the test scenes.

| Priors       | Results after running main_super_gaussian.py                                | Results after running evaluation.py |
|--------------|-----------------------------------------------------------------------------| ---------|
| GigaGAN      | [downloading link](https://uofi.box.com/s/cjqqlr0zfjw0ew02p1m5cm8ge6v73oow) | gigagan_prior folder in [link](https://uofi.box.com/s/cwjeo5sp6t2d81wcof0okqv6fn7aje8p) |
| VideoGigaGAN | [downloading link](https://uofi.box.com/s/xma4iqhirebmtzzj01s40jnjdg12unnk) | realbasicvsr_prior folder in [link](https://uofi.box.com/s/cwjeo5sp6t2d81wcof0okqv6fn7aje8p) |
| RealBasicVSR | [downloading link](https://uofi.box.com/s/lzakjip07upgullx7xunu6sausdydd95) | videogigagan_prior folder in [link](https://uofi.box.com/s/cwjeo5sp6t2d81wcof0okqv6fn7aje8p) |

2.1 Here is detailed information on the output file structure after running main_super_gaussian.py
```
--- [category_id]/[scene_id]
    --- 64x64               # low-res images
        --- 0000.png
        --- 0001.png
        ...
    --- gt/high_res_images   # high-res images
        --- 0000.png
        --- 0001.png
        ...
    --- step_1_upsampling    # intermediate results after running the first step of upsampling, either image upsampling or video upsampling
        --- 256x256           
            --- 0000.png
            --- 0001.png
            ...
    --- step_2_fitting_with_3dgs  # final results after running the second step of 3DDGS
        --- point_cloud
            --- iteration_0
            --- iteration_2000
                --- predicted              # renderings from upsampled 3DGS
                    --- 000.png
                    --- 001.png
                    ...
                point_cloud.ply            # upsampled 3DGS
                performance_novel.json     # reference based performance averaged within this scene
        --- cameras.json                 # camera extrinsics and intrinsics 
        --- input.ply                    # low-res 3DGS, you can also used color point cloud to initalize. 
    --- configs.json                     # configurations used for this scene   
    --- transforms.json                  # camera info used for this scene
    --- surface_pcd_131072_seed_0.ply    # low-res 3DGS, same as the one in input.ply (which is copied from this file)
```
2.2 Here is detailed information on the output file structure after running evaluation.py
```
--- evaluations
    --- gigagan_prior/realbasicvsr_prior/videogigagan_prior
        --- gt                  # pseudo gt images rendered from high-res 3DGS on novel trajectory traj_0 sampled from all scenes for evaluation
            --- XXX.png
            --- XXX.png
            ...
        --- pred                # rendering from 3d upsampled gaussians on novel trajectory traj_0 sampled from all scenes for evaluation
            --- XXX.png
            --- XXX.png
            ...
            
        --- gt_fid              # gt images in the original MVImgNet which are sampled from all scenes for evaluation on FID
            --- XXX.png
            --- XXX.png
            ...
```
3. We provide the quantitative results using the above cached upsampling images below. Note RealBasicVSR is a third-party method, which we newly benchmarked. 
```
| Priors       | LPIPS ↓ | NIQE  ↓ | FID  ↓ | 
|--------------|---------|---------|--------|
| GigaGAN      | 0.1522  | 7.65    | 27.04  |
| VideoGigaGAN | 0.1290  | 6.80    | 24.24  | 
| RealBasicVSR | 0.1924  | 7.58    | 41.40  |
```
Note some of the above numbers for GigaGAN and VideoGigaGAN are slightly better than the reported results in the paper, as we re-generated the above cached upsampled results in different dev environment used during our submission (some seeds might be different).


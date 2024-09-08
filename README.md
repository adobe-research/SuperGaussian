# SuperGaussian: Repurposing Video Models for 3D Super Resolution

Yuan Shen, Duygu Ceylan, Paul Guerrero, Zexiang Xu, Niloy J. Mitra, Shenlong Wang, and Anna Frühstück

ECCV 2024

[Paper](https://arxiv.org/abs/2406.00609) │ [Project Page](https://supergaussian.github.io/)

TLDR: Instead of image prior, we use video upsampling prior to achieve 3D upsampling on generic low-res 3D representations. 

In this codebase, we provide a SuperGaussian implementation on a third-parties video upsampling prior, [RealBasicVSR](https://github.com/ckkelvinchan/RealBasicVSR). 
Additionally, our codebase provides 3D upsampling evaluations on the MVImgNet testset given the upsampled image sequences from the image prior GigaGAN (image) and video prior Videogigagan (video).
Hopefully, this could help the community to reproduce our results and compare with other methods.

### Dependencies

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

# install REALBasicvsr in another conda env
conda create -n realbasicvsr python=3.8 -y
conda activate realbasicvsr
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html # it needs to be torch 1.7.1. For CuDA, it depends on your GPU compatibility.
pip install mmcv-full==1.5.0
pip install mmedit==0.15.0

conda activate super_gaussian_eccv24
```

#### 2. Install with Docker
```bash
docker pull yshen47/adobe_supergaussian:latest
docker run -it -v $(DATA_PATH):/mnt/data --shm-size=64g --gpus all yshen47/adobe_supergaussian:latest bash
cd /root
cd SuperGaussian_ECCV24
ln -s /mnt/data data
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

### Evaluation
1. To run inference on all our test scenes, you can use the following command. If you do not feel like running the inference, go to 2.
```bash
python main_super_gaussian.py    # change upsampling_prior variable in Line 23 to switch between different priors.
python evaluation.py             # change target variable in Line 94 to switch between different priors.
```
2. We provide the above evaluation results for all priors, if you hope to make an Apple-to-Apple comparison between our SuperGaussian using VideoGigaGAN with your method. 
You can download from this link to get all our inference results on GigaGAN, VideoGigaGAN and RealBasicVSR. You are able to access 3D upsampled gaussians 
and final 4x renderings with all poses in the test scenes.


3. We provide the quantitative results using the above cached upsampling images below. Note RealBasicVSR is a third-party method, which we newly benchmarked. 
```
| Priors       | LPIPS ↓ | NIQE  ↓ | FID  ↓ | 
|--------------|---------|---------|--------|
| GigaGAN      | 0.1522  | 7.65    | 27.04  |
| VideoGigaGAN | 0.1290  | 6.80    | 24.24  | 
| RealBasicVSR | 0.1924  | 7.58    | 41.40  |
```
Note some of the above numbers for GigaGAN and VideoGigaGAN are slightly better than the reported results in the paper, as we re-generated the above cached upsampled results in different dev environment used during our submission (some seeds might be different).
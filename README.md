# SuperGaussian: Repurposing Video Models for 3D Super Resolution

Yuan Shen, Duygu Ceylan, Paul Guerrero, Zexiang Xu, Niloy J. Mitra, Shenlong Wang, and Anna Frühstück

ECCV 2024

[Paper](https://arxiv.org/abs/2406.00609) │ [Project Page](https://supergaussian.github.io/)

TLDR: Instead of image prior, we use video upsampling prior to achieve 3D upsampling on generic low-res 3D representations. 

In this codebase, we provide a SuperGaussian implementation on a third-parties video upsampling prior, [RealBasicVSR](https://github.com/ckkelvinchan/RealBasicVSR). 
Additionally, our codebase provides our 3D upsampling evaluations on the MVImgNet test set given the upsampled image sequence from the image prior GigaGAN (image) and video prior Videogigagan (video).
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

1. Download the MVImgNet test set with 523 scenes. Each scene are grouped first by its category id and then scene id, defined by MVImgNet.
The test scenes are selected for its rich details, diversity and challenging scenarios. Within each scene folder, the directory are organized as follows:
```     
--- [category_id]/[scene_id]
    --- cam_extrinsics.pkl
    --- cam_intrinsics.pkl
    --- rgb.pkl
    --- xyz.pkl
    --- novel_trajectory_poses
        --- 0000.json
        --- 0001.json
        ...
    --- gt_rgb
        --- 0000.png
        --- 0001.png
        ...
    --- LR_131072_gaussian
        --- 0000.png
        --- 0001.png
        ...
    --- HR_131072_gaussian
        --- 0000.png
        --- 0001.png
        ...
        
    --- gigagan_image
        --- 0000.png
        --- 0001.png
        ...
    ... upsampled_rgb_videogigagan
        --- 0000.png
        --- 0001.png
        ...
```bash
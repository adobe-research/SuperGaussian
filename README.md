# SuperGaussian_ECCV24

Official codebase for SuperGaussian accepted to ECCV 2024. 

### Installment
```
conda create -n super_gaussian_eccv24 python=3.8 -y

conda activate super_gaussian_eccv24
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
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

### Inference

Run inference on MVImgNet testset with a third-party VSR prior, RealBasicVSR. Note, due to license issue, we cannot release one with VideoGigaGAN or GigaGAN checkpoints. 
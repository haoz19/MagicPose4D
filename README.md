<p align="center">

  <h2 align="center">MagicPose4D: Crafting Articulated Models <br>with Appearance and Motion Control</h2>
  <p align="center">
    <a href="https://haoz19.github.io/"><strong>Hao Zhang</strong></a><sup>1*</sup>
    ·  
    <a href="https://boese0601.github.io/"><strong>Di Chang</strong></a><sup>2*</sup>
    ·
    <a href="https://fangli333.github.io/"><strong>Fang Li</strong></a><sup>1</sup>
    ·
    <a href="https://www.ihp-lab.org/"><strong>Mohammad Soleymani</strong></a><sup>2</sup>
    ·
    <a href="https://vision.ai.illinois.edu/narendra-ahuja/"><strong>Narendra Ahuja</strong></a><sup>1</sup>
    <br>
    <sup>1</sup>University of Illinois Urbana-Champaign &nbsp;&nbsp;&nbsp; <sup>2</sup>University of Southern California
    <br>
    <sup>*</sup>Equal Contribution &nbsp;&nbsp;&nbsp; 
    <br>
    </br>
        <a href="https://arxiv.org/abs/2405.14017">
        <img src='https://img.shields.io/badge/arXiv-MagicPose4D-green' alt='Paper PDF'>
        </a>
        <a href='https://boese0601.github.io/magicpose4d/'>
        <img src='https://img.shields.io/badge/Project_Page-MagicPose4D-blue' alt='Project Page'></a>
        <!-- <a href='https://youtu.be/VPJe6TyrT-Y'>
        <img src='https://img.shields.io/badge/YouTube-MagicPose-rgb(255, 0, 0)' alt='Youtube'></a> -->
     </br>
    <table align="center">
        <img src="./figures/hiphop-1-humanoid.gif">
    </table>
</p>

*We introduce MagicPose4D, a novel framework for 4D generation providing more accurate and customizable 4D motion retargeting. We propose a dual-phase reconstruction process that initially uses accurate 2D and pseudo 3D supervision without skeleton constraints, and subsequently refines the model with skeleton constraints to ensure physical plausibility. We incorporate a novel Global-Local Chamfer loss function that aligns the overall distribution of mesh vertices with the supervision and maintains part-level alignment without additional annotations. Our method enables cross-category motion transfer using a kinematic-chain-based skeleton, ensuring smooth transitions between frames through dynamic rigidity and achieving robust generalization without the need for additional training.*

*For 3D reconstruction from monocular videos, please also check our previous work [S3O & LIMR](https://github.com/haoz19/LIMR)!*

*For 2D video motion retargeting and animation, please also check our previous work <a href="https://github.com/Boese0601/MagicDance">MagicPose</a>!*



## News
* **[2024.5.22]** Demo for the whole MagicPose4D pipeline is coming soon.
* **[2024.5.22]** Release MagicPose4D paper and project page.
* **[2024.5.22]** Release Python bindings for [Automatic-Rigging](https://github.com/haoz19/Automatic-Rigging).


## Install

* Please follow the repository: [Automatic-Rigging](https://github.com/haoz19/Automatic-Rigging) to install the package.
  ```
  git clone https://github.com/haoz19/Automatic-Rigging.git
  cd Automatic-Rigging
  pip install .
  ```

  This package is for calculating skinning weights and aligning the mesh with template skeletons.

* Install manifold remeshing:
  ```
  git clone --recursive git@github.com:hjwdzh/Manifold.git; cd Manifold; mkdir build; cd build; cmake .. -DCMAKE_BUILD_TYPE=Release;make -j8; cd ../../
  ```

* Install [threestudio](https://github.com/threestudio-project/threestudio) for image-to-3D generation
  
  See [installation.md](docs/installation.md) for additional information, including installation via Docker.
  
  The following steps have been tested on Ubuntu20.04.
  
  - You must have an NVIDIA graphics card with at least 6GB VRAM and have [CUDA](https://developer.nvidia.com/cuda-downloads) installed.
  - Install `Python >= 3.8`.
  - (Optional, Recommended) Create a virtual environment:
  
  ```sh
  python3 -m virtualenv venv
  . venv/bin/activate
  
  # Newer pip versions, e.g. pip-23.x, can be much faster than old versions, e.g. pip-20.x.
  # For instance, it caches the wheels of git packages to avoid unnecessarily rebuilding them later.
  python3 -m pip install --upgrade pip
  ```
  
  - Install `PyTorch >= 1.12`. We have tested on `torch1.12.1+cu113` and `torch2.0.0+cu118`, but other versions should also work fine.
  
  ```sh
  # torch1.12.1+cu113
  pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
  # or torch2.0.0+cu118
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```
  
  - (Optional, Recommended) Install ninja to speed up the compilation of CUDA extensions:
  
  ```sh
  pip install ninja
  ```
  
  - Install dependencies:
  
  ```sh
  pip install -r requirements.txt
  ```
  
  - (Optional) `tiny-cuda-nn` installation might require downgrading pip to 23.0.1
  
  - (Optional, Recommended) The best-performing models in threestudio use the newly-released T2I model [DeepFloyd IF](https://github.com/deep-floyd/IF), which currently requires signing a license agreement. If you would like to use these models, you need to [accept the license on the model card of DeepFloyd IF](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0), and login into the Hugging Face hub in the terminal by `huggingface-cli login`.

  

## Dual-phase 4D Reconstruction Module

* Generating 3D pseudo supervision via [zero-123](https://zero123.cs.columbia.edu/), also you can use [stable-123](https://stability.ai/stable-3d), [Magic-123](https://guochengqian.github.io/project/magic123/) or other image-to-3D methods.
  ```
  # Single image example:
  # Trainning:
  python launch.py --config configs/zero123.yaml --train --gpu 0 data.image_path=load/images/<input image>
  # Mesh Extraction:
  python launch.py --config outputs/zero123/<output_folder>/configs/parsed.yaml data.image_path=load/images/<input image> system.exporter.context_type=cuda --export --gpu 0 resume=outputs/zero123/<output_folder>/ckpts/last.ckpt  system.exporter_type=mesh-exporter system.geometry.isosurface_method=mc-cpu system.geometry.isosurface_resolution=256
  
  # Batch example: coming soon
  ```
  
* First-phase: Code coming soon
  
  For the first-phase we only want to get a sequcense of accurate 3D meshes, you can also consider other existing 4D recosntruction methods such as: [LIMR/S3O](https://github.com/haoz19/LIMR), [LASR](https://github.com/google/lasr), and [BANMo](https://github.com/facebookresearch/banmo) for reconstructing articulated objects from monocular videos.
* Second-phase: `<root>/PoseTransfer/batch_run.sh`
  
  This step ensures the skinning weights and skeleton are physically plausible.

## Cross-Category Pose Transfer Module

* Reference to our [demo](https://github.com/haoz19/MagicPose4D/blob/main/PoseTransfer/PoseTransfer_demo.ipynb).


## Citing
If you find our work useful, please consider citing:
```BibTeX
@misc{zhang2024magicpose4d,
      title={MagicPose4D: Crafting Articulated Models with Appearance and Motion Control}, 
      author={Hao Zhang and Di Chang and Fang Li and Mohammad Soleymani and Narendra Ahuja},
      year={2024},
      eprint={2405.14017},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```





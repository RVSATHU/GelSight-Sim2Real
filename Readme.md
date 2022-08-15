# Code repository of paper entitled "Bidirectional Sim-to-Real Transfer for GelSight Tactile Sensors with CycleGAN"

## Introduction

In this work, we show that bidirectional sim-real transfer for [GelSight](https://github.com/mcubelab/gelslim) -like sensors can be realized with [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). For Sim2Real, the transferred tactile images capture the non-ideal lighting conditions; for Real2Sim, the transferred images could produce more accurate depth maps.

Thanks to [GelSlim 3.0](https://github.com/mcubelab/gelslim) and [DIGIT](https://github.com/facebookresearch/digit-design), we can make GelSight sensors of our own. Please refer to them for sensor fabrication, calibration, depth reconstruction, etc.

If you find this project useful, please cite:
```
@ARTICLE{BidirSim2Real2022RAL,
  author={Chen, Weihang and Xu, Yuan and Chen, Zhenyang and Zeng, Peiyu and Dang, Renjun and Chen, Rui and Xu, Jing},
  journal={IEEE Robotics and Automation Letters},
  title={Bidirectional Sim-to-Real Transfer for GelSight Tactile Sensors With CycleGAN},
  year={2022},
  volume={7},
  number={3},
  pages={6187-6194},
  doi={10.1109/LRA.2022.3167064}}
```

## Self-collected dataset
### Simulation
Simulation-related code is in 'simulation_tacto' folder. In this simulator, depth maps are acquired from [Tacto](https://github.com/facebookresearch/tacto) and then post-processed by the [DoG](https://github.com/danfergo/gelsight_simulation) method. Then Phong's shading is performed to get the RGB tactile images.
### Download of full dataset
We uploaded the full dataset to Google Drive ([link](https://drive.google.com/file/d/130j4NyVsGifa_UO0478Aqdd8bVN9Ff8c/view?usp=sharing)).

In the compressed file, A is the real dataset while B is the simulation set. The images are named according to the object in contact. For each object, 1200 tactile images are collected both for Real and Sim set. Note that CycleGAN does not need training data to be paired, so the dataset can be collected through a simple procedure.

## Examples of Real2Sim depth reconstruction

Please run 'depth_reconstruction/calibrate_and_quantitatively_compare_depth.py'.

This script will first preform calibration and generate two LUTs. Then, depth maps will be reconstructed using the calibrated LUT. Depth maps from Real images and Real2Sim images will be compared to show the reconstruction error.

We would like to note that most of the code in this part is from [GelSlim 3.0](https://github.com/mcubelab/gelslim).

## CycleGAN Training and data generation using trained generators

In this research, we changed very little of CycleGAN. Please refer to the appendix file of pur paper for detailed training settings.
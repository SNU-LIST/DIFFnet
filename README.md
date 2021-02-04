# DIFFnet
* the code is for reconstructing diffusion model parameters from various diffusion gradient schemes and b-values using deep learning (DIFFnet).
* last update : 2020. 02. 04
* The source data for training can be shared to academic institutions. Request should be sent to snu.list.software@gmail.com. For each request, individual approval from our institutional review board is required (i.e. takes time)

# Reference

# Overview

<fig1 src="/Figure/fig1.png" width="300" height="300">
![figure 1](/Figure/fig1.png){: width="50" height="50"}

![figure 2](/Figure/fig2.png){: width="80" height="80"}

# Requirements 
* Python 3.7

* TensorFlow-gpu 1.15

* NVIDIA GPU (CUDA 10.0)

* MATLAB 2019a

# Data acquisition

* 3T MRI system (Tim Trio, SIEMENS, Erlangen, Germany) using a 32-channel phased-array head coil.

* DatasetDTI-A and DatasetNODDI-A were form below reference.
  * W. Jung et al., "Whole brain g-ratio mapping using myelin water imaging (MWI) and neurite orientation dispersion and density imaging (NODDI)," 
    NeuroImage, vol. 182, pp. 379-388, Nov. 2018.
    https://www.sciencedirect.com/science/article/pii/S1053811917308017

* DatasetDTI-A (b = 700 s/mm^2 with 32 directinos)

* DatasetDTI-B (b = 1000 s/mm^2 with 30 directions)

* DatasetNODDI-A (b = 300 s/mm^2 with 8 directions; b = 700 s/mm^2 with 32 directions; b = 2000 s/mm^2 with 64 directions) 

* DatasetNODDI-B (b = 300 s/mm^2 with 8 directions; b = 700 s/mm^2 with 30 directions; b = 2000 s/mm^2 with 60 directions) 

# Usage
### Simulation 

* Monte-Carlo diffusion simulation code to generate diffusion-weighted signals for training.

### Training

* The source code for training DIFFnet. Simulated data from Monte-Carlo diffusion simulation has to be required.

### Evaluation

* The source code for evaluation of the trained networks.
* in-vivo data and simulated data can be evaluated both.
* networks generate diffusion model parameters.


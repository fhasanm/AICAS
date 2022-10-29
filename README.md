# AICAS - Artificially Intelligent Collision Avoidance System 

Official repository of AICAS


## Installation
### Install Anaconda
We recommend using Anaconda.
The installation is described on the following page:\
https://docs.anaconda.com/anaconda/install/linux/

### Install Required Packages
```sh
conda env create -f environment.yml
```

### Activate Environment
```sh
conda activate sit
```

### Walkthrough

- GCN.py contains the Graph Convolution Network for self-intention prediction.
- models.py contains the Neighbour Trajectory Prediction model
- feeder.py contains code to preprocess Brain4Cars, BLVD and NGSIM dataset
- main.py contains the script to train-val-test Neighbour Trajectory Prediction (currently preprocessing script for GCN not ready yet)


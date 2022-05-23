# SqRL
This GitHub repository contains the implementation of the Square Rotational Layer for obtaining rotational invariance. SqRL was developed during the 6<sup>th</sup> semester Computer Science course at Aalborg University as a part of the Bachelor Project.

## Repo Structure
The networks folder contains all files for running our tests. 

### Datasets.ipynb
This file contains methods for preprocessing of the data sets. Running each code block in order will yield a folder containing the MNIST-10, Fashion-MNIST, SVHN and CIFAR-10 datasets with no augmentation (used for polar tests) and with SqRL transformation (used for SqRL tests).

### Layers
This folder contains the PyTorch network modules implementations. The SqRL is a module implementation that is not used during our tests as we preprocess the transformation. The Max module aggregates over rows assuming a PyTorch structure of the batch. The Polar module transforms images into polar representation assuming a PyTorch batch structure.


## Running the Networks

To run the test use the following command when placed in the network folder:

`python main.py -i <iterations> -d <data set> -m <model>`

For example to run our SqRL model on the MNIST-10 data set with 100 different samples:

`python main.py -i 100 -d MNIST-SQRL -m sqrl`

All used commands are as follows:

`python main.py -i 100 -d MNIST-SQRL -m sqrl`\
`python main.py -i 100 -d FMNIST-SQRL -m sqrl`\
`python main.py -i 100 -d SVHN-SQRL -m sqrl`\
`python main.py -i 100 -d CIFAR-10-SQRL -m sqrl`\
`python main.py -i 100 -d MNIST-SQRL -m polar`\
`python main.py -i 100 -d FMNIST -m polar`\
`python main.py -i 100 -d SVHN -m polar`\
`python main.py -i 100 -d CIFAR-10 -m polar`

# Dependencies
- Python 3.9.5
- CUDA 11.5
- Jupyter

## Python libraries
- Torch 1.10.0
- Torchvision 0.11.1
- Numpy 1.21.3
- OpenCV 4.5.5.62

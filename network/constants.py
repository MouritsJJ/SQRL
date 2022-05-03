# Dictionary for dataset paths
dataset_paths = {
    # Root of the folder containing the CIFAR images
    'CIFAR-10': 
    [
        'C:/Users/p5aau/Desktop/CIFAR-10 Rot Datasets/train',
        'C:/Users/p5aau/Desktop/CIFAR-10 Rot Datasets/test',
        'C:/Users/p5aau/Desktop/CIFAR-10 Rot Datasets/CIFAR-10-45',
        'C:/Users/p5aau/Desktop/CIFAR-10 Rot Datasets/CIFAR-10-90',
        'C:/Users/p5aau/Desktop/CIFAR-10 Rot Datasets/CIFAR-10-360'
    ],

    # Root of the folder containing the MNIST images
    'MNIST':
    [
        'C:/Users/p5aau/Desktop/MNIST Rot Datasets/train',
        'C:/Users/p5aau/Desktop/MNIST Rot Datasets/test',
        'C:/Users/p5aau/Desktop/MNIST Rot Datasets/MNIST-45',
        'C:/Users/p5aau/Desktop/MNIST Rot Datasets/MNIST-90',
        'C:/Users/p5aau/Desktop/MNIST Rot Datasets/MNIST-360'
    ],

    # Root of the folder containing the Fashion MNIST images
    'FMNIST':
    [
        'C:/Users/p5aau/Desktop/FMNIST Rot Datasets/train',
        'C:/Users/p5aau/Desktop/FMNIST Rot Datasets/test',
        'C:/Users/p5aau/Desktop/FMNIST Rot Datasets/FMNIST-45',
        'C:/Users/p5aau/Desktop/FMNIST Rot Datasets/FMNIST-90',
        'C:/Users/p5aau/Desktop/FMNIST Rot Datasets/FMNIST-360'
    ],

    # Root of the folder containing the SVHN images
    'SVHN':
    [
        'C:/Users/p5aau/Desktop/SVHN Rot Datasets/train',
        'C:/Users/p5aau/Desktop/SVHN Rot Datasets/test',
        'C:/Users/p5aau/Desktop/SVHN Rot Datasets/SVHN-45',
        'C:/Users/p5aau/Desktop/SVHN Rot Datasets/SVHN-90',
        'C:/Users/p5aau/Desktop/SVHN Rot Datasets/SVHN-360'
    ]
}

# Channels in training images
image_channels = 3

# Batch size of training data
batch_size = 128

# Number of workers in dataloader
num_workers = 4

# Numbers of classes to classify
num_classes = 10

# Boolean to determine if weights should be initialises
init_weights = True

# Momentum for optimizer
momentum = 0.9

# Learning rate for optimizer
lr = 0.05

# Weight decay for optimizer
weight_decay = 1e-5

# Factor for learning rate scheduler
factor = 0.2

# Patience for learning rate scheduler
patience = 2

# Cooldown for learning rate scheduler
cooldown = 0

# Minimum learning rate for learning rate scheduler
min_lr = 1e-6

# Patience for training
train_patience = 5

# linear constant
linear = 512
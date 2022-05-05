# Dictionary for dataset paths
dataset_paths = {
    # Root of the folder containing the CIFAR images
    'CIFAR-10': 
    [
        'data/cifar-10/train_x.npy',
        'data/cifar-10/train_y.npy',
        'data/cifar-10/test_x.npy',
        'data/cifar-10/test_y.npy',
        'data/cifar-10/test_x_45.npy',
        'data/cifar-10/test_x_90.npy',
        'data/cifar-10/test_x_360.npy'
    ],

    # Root of the folder containing the MNIST images
    'MNIST':
    [
        'data/mnist/train_x.npy',
        'data/mnist/train_y.npy',
        'data/mnist/test_x.npy',
        'data/mnist/test_y.npy',
        'data/mnist/test_x_45.npy',
        'data/mnist/test_x_90.npy',
        'data/mnist/test_x_360.npy'
    ],

    # Root of the folder containing the Fashion MNIST images
    'FMNIST':
    [
        'data/fmnist/train_x.npy',
        'data/fmnist/train_y.npy',
        'data/fmnist/test_x.npy',
        'data/fmnist/test_y.npy',
        'data/fmnist/test_x_45.npy',
        'data/fmnist/test_x_90.npy',
        'data/fmnist/test_x_360.npy'
    ],

    # Root of the folder containing the SVHN images
    'SVHN':
    [
        'data/svhn/train_x.npy',
        'data/svhn/train_y.npy',
        'data/svhn/test_x.npy',
        'data/svhn/test_y.npy',
        'data/svhn/test_x_45.npy',
        'data/svhn/test_x_90.npy',
        'data/svhn/test_x_360.npy'
    ],

    # Root of the folder containing the CIFAR images
    'CIFAR-10-SQRL': 
    [
        'data/cifar-10/sqrl_train_x.npy',
        'data/cifar-10/train_y.npy',
        'data/cifar-10/sqrl_test_x.npy',
        'data/cifar-10/test_y.npy',
        'data/cifar-10/sqrl_test_x_45.npy',
        'data/cifar-10/sqrl_test_x_90.npy',
        'data/cifar-10/sqrl_test_x_360.npy'
    ],

    # Root of the folder containing the MNIST images
    'MNIST-SQRL':
    [
        'data/mnist/sqrl_train_x.npy',
        'data/mnist/train_y.npy',
        'data/mnist/sqrl_test_x.npy',
        'data/mnist/test_y.npy',
        'data/mnist/sqrl_test_x_45.npy',
        'data/mnist/sqrl_test_x_90.npy',
        'data/mnist/sqrl_test_x_360.npy'
    ],

    # Root of the folder containing the Fashion MNIST images
    'FMNIST-SQRL':
    [
        'data/fmnist/sqrl_train_x.npy',
        'data/fmnist/train_y.npy',
        'data/fmnist/sqrl_test_x.npy',
        'data/fmnist/test_y.npy',
        'data/fmnist/sqrl_test_x_45.npy',
        'data/fmnist/sqrl_test_x_90.npy',
        'data/fmnist/sqrl_test_x_360.npy'
    ],

    # Root of the folder containing the SVHN images
    'SVHN-SQRL':
    [
        'data/svhn/sqrl_train_x.npy',
        'data/svhn/train_y.npy',
        'data/svhn/sqrl_test_x.npy',
        'data/svhn/test_y.npy',
        'data/svhn/sqrl_test_x_45.npy',
        'data/svhn/sqrl_test_x_90.npy',
        'data/svhn/sqrl_test_x_360.npy'
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
patience = 3

# Cooldown for learning rate scheduler
cooldown = 0

# Minimum learning rate for learning rate scheduler
min_lr = 1e-6

# Patience for training
train_patience = 10

# linear constant
linear = 640
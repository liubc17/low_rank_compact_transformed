# low_rank_compact_transformed
A Novel Compact Design of Convolutional Layers with Spatial Transformation towards Lower-rank Representation for Image Classification

Code for paper 'A Novel Compact Design of Convolutional Layers with Spatial Transformation towards Lower-rank Representation for Image Classification'.

python version 3.7

torch version 1.9

numpy version 1.21

Our low-rank compact transformed design of convolutional layer is in STN.py

Compressed networks are defined in vgg_tucker.py, CMT_tucker.py, LeNet_tucker.py, ResNet_imageNet_tucker.py and ResNet_tucker.py

For training different datasets, choose from train_mnist.py, train_cifar10.py, train_cifar100.py and train_imagenet.py

For training, the user must manually declare the network architecture in '--arch' or '-a'. The channel compression rate, compressed kernel size and kernel size can be declare optionally, default 0.5, 3, 3, respectively. 

For example, if the user wants to train CIFAR10 via compressed ResNet32 with channel compression rate 0.75, compressed kernel size 3 and kernel size 5, the user should run train_cifar10.py and declare '-a Tuckerresnet32 --com_rate 0.75 --com_ker_size 3 --ker_size 5'.

To finetune a network on ImageNet from a pre-trained model, run load_decomposition.py. Select a pre-trained network and a channel compression rate. Then it will generate a decomposed '.pth' file. Then, while running train_imagenet.py, expect for declaring the network architecture, channel compression rate, compressed kernel size and kernel size, the user should also declare '--pre_path' with the decomposed '.pth' file.

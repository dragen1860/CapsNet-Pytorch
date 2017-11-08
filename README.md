# CapsNet-Pytorch
Pytorch version of Hinton's paper: Dynamic Routing Between Capsules

## Status

- Currently we train our model for 30 epochs, which means it is potential if more epochs are used to train
- We don not use reconstruction loss now, and will add it later
- The critical part of code is well commented with each dimension changes, which means you can follow the comments to understand the routing mechnism

## TODO
- add reconstruction loss
- test on more convincing dataset, such as ImagetNet

## About me
> I'm a Research Assistant @ National University of Singapre, before joinging NUS, I was a first-year PhD candidate in Zhejiang University and then quitted.
Contact me with email: dcslong@nus.edu.sg or wechat: dragen1860

# Usage

## Step 1. Install Conda, CUDA, cudnn and Pytorch
>conda install pytorch torchvision cuda80 -c soumith

## Step 2. Clone the repository to local
>git clone https://github.com/dragen1860/CapsNet-Pytorch.git
cd CapsNet-Pytorch

## Step 3. Train CapsNet on MNIST

1. please modify the variable `glo_batch_size = 125` to appropriate size according to your GPU memory size.
2. run
>$ python main.py
3. turn on tensorboard version of pytorch
>$ tensorboard --logdir runs 

4. OR you can comment the part of train code and test its performance with pretrained model `mdl` file.

# Results

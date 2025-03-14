#!/bin/sh
python3 main.py --dataset cifar10 \
    --data_path /gpfs/mariana/home/yukoba/pytorch-cifar10/cifar-10-batches-py   \
    --arch mobilenet_quan_cifar10 --save_path /gpfs/mariana/home/yukoba/Neural_Network_Weight_Attack/save/TEST  \
    --resume mobilenet_cifar10.pth \
    --test_batch_size 256 --workers 8 --ngpu 0 \
    --print_freq 50 \
    --reset_weight --bfa --n_iter 20 --k_top 10 \
    --attack_sample_size 128

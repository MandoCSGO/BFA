#!/bin/sh
python3 main.py --dataset cifar100 \
    --data_path /gpfs/mariana/home/yukoba/pytorch-cifar100/data/cifar-100-python   \
    --arch vgg16_quan --save_path /gpfs/mariana/home/yukoba/Neural_Network_Weight_Attack/save/TEST  \
    --resume vgg16_cifar100 \
    --test_batch_size 256 --workers 8 --ngpu 0 \
    --print_freq 50 \
    --reset_weight --bfa --n_iter 20 --k_top 10 \
    --attack_sample_size 128

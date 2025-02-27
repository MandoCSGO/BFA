#!/bin/sh
python3 main.py --dataset cifar10 \
    --data_path /gpfs/mariana/home/yukoba/pytorch-cifar10/cifar-10-batches-py   \
    --arch vgg16_quan_hardened --save_path /gpfs/mariana/home/yukoba/Neural_Network_Weight_Attack/save/0.05  \
    --resume hardened_model-deepvigor-None-0.05.pth \
    --test_batch_size 256 --workers 8 --ngpu 1 \
    --print_freq 50 \
    --reset_weight --bfa --n_iter 256 --k_top 20 \
    --attack_sample_size 128

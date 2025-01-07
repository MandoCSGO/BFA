#!/usr/bin/env sh

############### Host   ##############################
HOST=$(hostname)
echo "Current host is: $HOST"

# Automatic check the host and configure
case $HOST in
"alpha")
    PYTHON="/home/yukoba/python3.11-venv" # python environment path
    TENSORBOARD='/home/yukoba/python3.11-venv/lib/python3.11/site-packages/tensorboard' # tensorboard environment path
    data_path='/home/yukoba/pytorch-cifar10' # dataset path
    ;;
esac

DATE=`date +%Y-%m-%d`

if [ ! -d "$DIRECTORY" ]; then
    mkdir ./save/${DATE}/
fi

############### Configurations ########################
enable_tb_display=false # enable tensorboard display
model=vgg16 
dataset=cifar10
test_batch_size=256

attack_sample_size=128 # number of data used for BFA
n_iter=20 # number of iteration to perform BFA
k_top=10 # only check k_top weights with top gradient ranking in each layer

save_path=/home/yukoba/Neural_Network_Weight_Attack/save/${DATE}/${dataset}_${model}
tb_path=./save/${DATE}/${dataset}_${model}_${epochs}_${optimizer}_${quantize}/tb_log  #tensorboard log path

############### Neural network ############################
{
python3 main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ${save_path}  \
    --test_batch_size ${test_batch_size} --workers 8 --ngpu 2 \
    --print_freq 50 \
    --reset_weight --bfa --n_iter ${n_iter} --k_top ${k_top} \
    --attack_sample_size ${attack_sample_size}
} &
############## Tensorboard logging ##########################
{
if [ "$enable_tb_display" = true ]; then 
    sleep 30 
    wait
    $TENSORBOARD --logdir $tb_path  --port=6006
fi
} &
{
if [ "$enable_tb_display" = true ]; then
    sleep 45
    wait
    case $HOST in
    "Hydrogen")
        firefox http://0.0.0.0:6006/
        ;;
    "alpha")
        google-chrome http://0.0.0.0:6006/
        ;;
    esac
fi 
} &
wait

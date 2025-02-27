#!/usr/bin/env sh

############### Configurations ########################
# Set paths directly without hostname check
PYTHON="/gpfs/mariana/home/yukoba/python3.11-venv" # python environment path
TENSORBOARD='/gpfs/mariana/home/yukoba/python3.11-venv/lib/python3.11/site-packages/tensorboard' # tensorboard environment path
data_path='/gpfs/mariana/home/yukoba/pytorch-cifar10' # dataset path

# Validate dataset path
if [ -z "$data_path" ]; then
    echo "Error: data_path is not set. Exiting..."
    exit 1
fi

# Current date for directory naming
DATE=$(date +%Y-%m-%d)

# Save directory setup
save_directory=./save/${DATE}/
if [ ! -d "$save_directory" ]; then
    mkdir -p "$save_directory"
fi

# Other configurations
enable_tb_display=false # enable tensorboard display
model=mobilenet_v2 
dataset=cifar10
test_batch_size=256

attack_sample_size=128 # number of data used for BFA
n_iter=20 # number of iteration to perform BFA
k_top=10 # only check k_top weights with top gradient ranking in each layer

save_path=/gpfs/mariana/home/yukoba/Neural_Network_Weight_Attack/save/${DATE}/${dataset}_${model}
tb_path=${save_directory}${dataset}_${model}_tb_log  # tensorboard log path

############### Neural network ############################
{
python3 main.py --dataset ${dataset} \
    --data_path ${data_path}   \
    --arch ${model} --save_path ${save_path}  \
    --test_batch_size ${test_batch_size} --workers 8 \
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
    google-chrome http://0.0.0.0:6006/ &
fi 
} &
wait


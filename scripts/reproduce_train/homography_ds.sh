#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"
GFNET_DIR=$(readlink -f "${PROJECT_DIR}/../..")        # GFNet root dir

# conda activate loftr
export PYTHONPATH="${GFNET_DIR}:${PROJECT_DIR}:${PYTHONPATH}"
cd $PROJECT_DIR

TRAIN_IMG_SIZE=448
data_cfg_path="configs/data/homography_trainval_${TRAIN_IMG_SIZE}.py"
main_cfg_path="configs/loftr/outdoor/loftr_ds_dense.py"

n_nodes=1
n_gpus_per_node=4
torch_num_workers=4
batch_size=5
pin_memory=true
train_dataset="vis_ir_drone"
test_dataset="vis_ir_drone"
exp_name="homo-ds-${train_dataset}-${TRAIN_IMG_SIZE}-bs=$(($n_gpus_per_node * $n_nodes * $batch_size))"

#-m debugpy --listen localhost:6666 --wait-for-client
    python \
    -m train_homography \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --exp_name=${exp_name} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} --pin_memory=${pin_memory} \
    --check_val_every_n_epoch=5 \
    --log_every_n_steps=1 \
    --flush_logs_every_n_steps=1 \
    --limit_val_batches=1. \
    --num_sanity_val_steps=10 \
    --benchmark=True \
    --ckpt_path=${GFNET_DIR}/third_party/LoFTR_homography/checkpoints/outdoor_ds.ckpt \
    --train_dataset=${train_dataset} \
    --test_dataset=${test_dataset} \
    --mode=train

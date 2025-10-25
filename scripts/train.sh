#!/bin/bash

EXP_NAME='peacap'
TIME_START=$(date "+%d-%m-%Y_%H:%M:%S")
LOG_FOLDER=logs
SUB_FOLDER=TRAIN/SAMPLE
SAVE_FILE=${LOG_FOLDER}/${SUB_FOLDER}/${EXP_NAME}
mkdir -p $SAVE_FILE 

TRAIN_LOG_FILE="$LOG_FOLDER/${SUB_FOLDER}/${EXP_NAME}/TRAINING_${TIME_START}.log"

# MODEL CONFIGURATION
model_path="models.peacap"
ext_path="ext_data/ext_memory_lvis.pkl"
input_image_resize=224
bs=6
accum_grad_iters=1

CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node 1 ./train.py \
    --model_path ${model_path} \
    --input_image_resize ${input_image_resize} \
    --ext_path ${ext_path} \
    --annotation_file_for_train annotations/captions_train2014_sampled.json \
    --out_dir results/${SUB_FOLDER}/${EXP_NAME} \
    --bs ${bs} \
    --accum_grad_iters ${accum_grad_iters} \
    --log_folder $SAVE_FILE \
    --low_resource false \
    |& tee -a  ${TRAIN_LOG_FILE}

#!/bin/bash

EXP_NAME='evcap2'
TIME_START=$(date "+%d-%m-%Y_%H:%M:%S")
LOG_FOLDER=logs
SUB_FOLDER=TRAIN/SAMPLE
SAVE_FILE=${LOG_FOLDER}/${SUB_FOLDER}/${EXP_NAME}
mkdir -p $SAVE_FILE 

TRAIN_LOG_FILE="$LOG_FOLDER/${SUB_FOLDER}/${EXP_NAME}/TRAINING_${TIME_START}.log"

# MODEL CONFIGURATION
model_path="models.evcap"
ext_path="ext_data/ext_memory_lvis.pkl"
input_image_resize=224
bs=6
accum_grad_iters=1

CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node 1 ./train_evcap_bert_patch.py \
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



# --model_path models.evcap_bert_selected_patch_analysis_ver3_big_vector --device cuda:1 --input_image_resize 900 --bs 2 --accum_grad_iters 32 --ext_path ext_data/sample_10_categories/ext_memory_with_32_embeddings.pkl --annotation_file_for_train annotations/captions_train2014_10_categories_selected_patch.json

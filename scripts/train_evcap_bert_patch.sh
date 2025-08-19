#!/bin/bash

EXP_NAME='evcap_bert_patch_analysis_ver2_big_vector'
DATA_SET_TYPE='10'
TIME_START=$(date "+%Y-%m-%d-%H-%M-%S")
LOG_FOLDER=logs
SUB_FOLDER=TRAIN/SAMPLED_${DATA_SET_TYPE}_CATEGORIES
SAVE_FILE=${LOG_FOLDER}/${SUB_FOLDER}/${EXP_NAME}
mkdir -p $SAVE_FILE 

TRAIN_LOG_FILE="$LOG_FOLDER/${SUB_FOLDER}/${EXP_NAME}/TRAINING_${TIME_START}.log"

# MODEL CONFIGURATION
model_path="models.evcap_bert_patch_analysis_ver2_big_vector"
ext_path="ext_data/sample_100_categories/ext_memory_with_32_embeddings.pkl"
input_image_resize=900
bs=2
accum_grad_iters=32

if [ "$DATA_SET_TYPE" = "100" ]; then
    CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node 2 ./train_evcap_bert_patch.py \
        --model_path ${model_path} \
        --input_image_resize ${input_image_resize} \
        --ext_path ${ext_path} \
        --annotation_file_for_train annotations/captions_train2014_in_scope.json \
        --out_dir results/${SUB_FOLDER}/${EXP_NAME} \
        --bs ${bs} \
        --accum_grad_iters ${accum_grad_iters} \
        --log_folder $SAVE_FILE \
        --low_resource false \
        |& tee -a  ${TRAIN_LOG_FILE}

elif [ "$DATA_SET_TYPE" = "10" ]; then
    CUDA_VISIBLE_DEVICES="1" torchrun --nproc_per_node 1 ./train_evcap_bert_patch.py \
        --model_path ${model_path} \
        --input_image_resize ${input_image_resize} \
        --ext_path ${ext_path} \
        --annotation_file_for_train annotations/captions_train2014_10_categories_selected_patch.json \
        --out_dir results/${SUB_FOLDER}/${EXP_NAME} \
        --bs ${bs} \
        --accum_grad_iters ${accum_grad_iters} \
        --log_folder $SAVE_FILE \
        --low_resource false \
        |& tee -a  ${TRAIN_LOG_FILE}

elif [ "$DATA_SET_TYPE" = "ALL" ]; then
    CUDA_VISIBLE_DEVICES="1" torchrun --nproc_per_node 1 ./train_evcap_bert_patch.py \
        --model_path ${model_path} \
        --input_image_resize ${input_image_resize} \
        --ext_path ${ext_path} \
        --annotation_file_for_train annotations/captions_train2014.json \
        --out_dir results/${SUB_FOLDER}/${EXP_NAME} \
        --bs ${bs} \
        --accum_grad_iters ${accum_grad_iters} \
        --log_folder $SAVE_FILE \
        --low_resource false \
        |& tee -a  ${TRAIN_LOG_FILE}
else
    echo "Invalid DATA_SET_TYPE. Please set it to '10', '100' or 'ALL'."
    exit 1
fi

# --model_path models.evcap_bert_selected_patch_analysis_ver3_big_vector --device cuda:1 --input_image_resize 900 --bs 2 --accum_grad_iters 32 --ext_path ext_data/sample_10_categories/ext_memory_with_32_embeddings.pkl --annotation_file_for_train annotations/captions_train2014_10_categories_selected_patch.json

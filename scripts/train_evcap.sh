#!/bin/bash

EXP_NAME='evcap'
TIME_START=$(date "+%d-%m-%Y_%H:%M:%S")
SUB_FOLDER=TRAIN/SAMPLE
SAVE_FILE=results/${SUB_FOLDER}/${EXP_NAME}
TRAIN_LOG_FILE=log/${SUB_FOLDER}/${EXP_NAME}/${TIME_START}.log
mkdir -p results/${SUB_FOLDER}/${EXP_NAME}
mkdir -p log/${SUB_FOLDER}/${EXP_NAME}

CUDA_VISIBLE_DEVICES="1" torchrun --nproc_per_node 1 ./train_evcap.py \
    --out_dir $SAVE_FILE \
    --annotation_file_for_train annotations/captions_train2014_sampled.json \
    --log_folder $SAVE_FILE \
    |& tee -a  ${TRAIN_LOG_FILE}

    # --model_path models.evcap \
    # --ext_path ext_data/sample_10_categories/ext_memory_original_format.pkl \
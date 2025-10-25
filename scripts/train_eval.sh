#!/bin/bash

# TRAINING
EXP_NAME='peacap_cross_att'
TIME_START=$(date "+%d-%m-%Y_%H:%M:%S")
LOG_FOLDER=logs
SUB_FOLDER=ABLATION_STUDY
SAVE_FILE=${LOG_FOLDER}/TRAIN/${SUB_FOLDER}/${EXP_NAME}
mkdir -p $SAVE_FILE 

TRAIN_LOG_FILE="$LOG_FOLDER/TRAIN/${SUB_FOLDER}/${EXP_NAME}/TRAINING_${TIME_START}.log"

#######################################################################################
# MODEL CONFIGURATION (Need to be adjusted)
model_path="models.ablation_study.peacap_cross_att"
ext_path="ext_data/ext_memory_lvis.pkl"
input_image_resize=680  
bs=1
accum_grad_iters=1
topn=9
SEED=42
#######################################################################################

CUDA_VISIBLE_DEVICES="1" torchrun --nproc_per_node 1 ./train.py \
    --model_path ${model_path} \
    --input_image_resize ${input_image_resize} \
    --ext_path ${ext_path} \
    --annotation_file_for_train annotations/captions_train2014_sampled.json \
    --out_dir results/TRAIN/${SUB_FOLDER}/${EXP_NAME} \
    --bs ${bs} \
    --log_folder $SAVE_FILE \
    --topn ${topn} \
    --random_seed ${SEED} \
    |& tee -a  ${TRAIN_LOG_FILE} \

    #  --accum_grad_iters ${accum_grad_iters} \

# EVALUATION
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER/..

DEVICE="1"
NOCAPS_OUT_PATH=results/EVAL/VAL/${SUB_FOLDER}/${EXP_NAME}

LOG_FOLDER=logs/EVAL/VAL/${SUB_FOLDER}/${EXP_NAME}
mkdir -p $LOG_FOLDER

NOCAPS_LOG_FILE="$LOG_FOLDER/NOCAPS_${TIME_START}.log"

##########################################
# MODEL CONFIGURATION (Need to be adjusted)
ckpt=results/TRAIN/${SUB_FOLDER}/${EXP_NAME}/final_result_000.pt

##########################################

python -u eval.py \
    --model_path ${model_path} \
    --path_of_val_datasets ./data/coco/coco2014/annotations/captions_val2014_sampled_005_fixed_format.json \
    --image_size ${input_image_resize} \
    --device cuda:$DEVICE \
    --name_of_datasets coco_val2014 \
    --out_path=$NOCAPS_OUT_PATH \
    --ckpt ${ckpt} \
    --ext_data_path ${ext_path} \
    --topn ${topn} \
    --log_folder ${LOG_FOLDER} \
    --random_seed ${SEED} \
    |& tee -a  ${NOCAPS_LOG_FILE}
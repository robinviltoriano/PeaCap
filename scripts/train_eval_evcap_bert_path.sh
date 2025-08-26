#!/bin/bash

# TRAINING
EXP_NAME='evcap_bert_patch_analysis_ver5_dot_2'
TIME_START=$(date "+%d-%m-%Y_%H:%M:%S")
LOG_FOLDER=logs
SUB_FOLDER=SAMPLE
SAVE_FILE=${LOG_FOLDER}/TRAIN/${SUB_FOLDER}/${EXP_NAME}
mkdir -p $SAVE_FILE 

TRAIN_LOG_FILE="$LOG_FOLDER/TRAIN/${SUB_FOLDER}/${EXP_NAME}/TRAINING_${TIME_START}.log"

# MODEL CONFIGURATION
model_path="models.evcap_bert_patch_analysis_ver5"
ext_path="ext_data/ext_memory_lvis.pkl"
input_image_resize=680  
bs=1
accum_grad_iters=1
topn=9

CUDA_VISIBLE_DEVICES="0,1" torchrun --nproc_per_node 2 ./train_evcap_bert_patch.py \
    --model_path ${model_path} \
    --input_image_resize ${input_image_resize} \
    --ext_path ${ext_path} \
    --annotation_file_for_train annotations/captions_train2014_sampled.json \
    --out_dir results/TRAIN/${SUB_FOLDER}/${EXP_NAME} \
    --bs ${bs} \
    --accum_grad_iters ${accum_grad_iters} \
    --log_folder $SAVE_FILE \
    --low_resource false \
    --topn ${topn} \
    |& tee -a  ${TRAIN_LOG_FILE} \
&& {
    # EVALUATION
    SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
    cd $SHELL_FOLDER/..

    DEVICE="0"
    NOCAPS_OUT_PATH=results/EVAL/VAL/${SUB_FOLDER}/${EXP_NAME}

    LOG_FOLDER=logs/EVAL/VAL/${SUB_FOLDER}/${EXP_NAME}
    mkdir -p $LOG_FOLDER

    NOCAPS_LOG_FILE="$LOG_FOLDER/NOCAPS_${TIME_START}.log"

    ##########################################
    # MODEL CONFIGURATION (Need to be adjusted)
    ckpt=results/TRAIN/${SUB_FOLDER}/${EXP_NAME}/final_result_000.pt

    ##########################################
    # Dataset Adjustments
    # path_of_val_datasets

    python -u eval_evcap_bert_patch.py \
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
    |& tee -a  ${NOCAPS_LOG_FILE}

}


# --model_path models.evcap_bert_patch_analysis_ver5 --input_image_resize 680 --ext_path ext_data/result/embeddings_32/ext_memory_lvis_distilled_with_img_id.pkl --annotation_file_for_train annotations/captions_train2014_sampled.json --bs 1 --accum_grad_iters 1 --low_resource false --topn 26


# --image_size 900 --model_path models.evcap_bert_patch_analysis_ver2_big_vector --path_of_val_datasets ./data/coco/karpathy/captions_testKarpathy_10_categories_val_reference_selected_patch.json --ckpt results/TRAIN/SAMPLED_10_CATEGORIES/PATCH_ANALYSIS/SELECTED_PATCHES/evcap_bert_selected_patch_analysis_ver2_big_vector/final_result_000.pt --ext_data_path ext_data/sample_10_categories/ext_memory_with_32_embeddings_and_selected_patch.pkl
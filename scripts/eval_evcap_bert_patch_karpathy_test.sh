SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER/..

EXP_NAME=$1
DEVICE=$2
SUB_FOLDER=EVAL/VAL/SAMPLE
NOCAPS_OUT_PATH=results/${SUB_FOLDER}/$EXP_NAME

TIME_START=$(date "+%d-%m-%Y_%H:%M:%S")
LOG_FOLDER=logs/${SUB_FOLDER}/${EXP_NAME}
mkdir -p $LOG_FOLDER

NOCAPS_LOG_FILE="$LOG_FOLDER/NOCAPS_${TIME_START}.log"

##########################################
# MODEL CONFIGURATION (Need to be adjusted)
model_path="models.evcap"
input_image_resize=224
ckpt="results/TRAIN/SAMPLE/evcap3/final_result_000.pt"
ext_data_path="ext_data/ext_memory_lvis.pkl"

##########################################
# Dataset Adjustments
# path_of_val_datasets

python -u eval_evcap_bert_patch.py \
--model_path ${model_path} \
--path_of_val_datasets ./data/coco/coco2014/annotations/captions_val2014_sampled_fixed_format.json \
--image_size ${input_image_resize} \
--device cuda:$DEVICE \
--name_of_datasets coco_val2014 \
--out_path=$NOCAPS_OUT_PATH \
--ckpt ${ckpt} \
--ext_data_path $ext_data_path \
--log_folder $LOG_FOLDER \
|& tee -a  ${NOCAPS_LOG_FILE}

# --image_size 900 --model_path models.evcap_bert_patch_analysis_ver2_big_vector --path_of_val_datasets ./data/coco/karpathy/captions_testKarpathy_10_categories_val_reference_selected_patch.json --ckpt results/TRAIN/SAMPLED_10_CATEGORIES/PATCH_ANALYSIS/SELECTED_PATCHES/evcap_bert_selected_patch_analysis_ver2_big_vector/final_result_000.pt --ext_data_path ext_data/sample_10_categories/ext_memory_with_32_embeddings_and_selected_patch.pkl
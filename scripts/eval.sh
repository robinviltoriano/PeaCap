SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)
cd $SHELL_FOLDER/..

EXP_NAME=$1
DEVICE=$2
SUB_FOLDER=FULL_DATA
NOCAPS_OUT_PATH=results/${SUB_FOLDER}/$EXP_NAME

TIME_START=$(date "+%d-%m-%Y_%H:%M:%S")
LOG_FOLDER=logs/EVAL/VAL/${SUB_FOLDER}/${EXP_NAME}
mkdir -p $LOG_FOLDER

NOCAPS_LOG_FILE="$LOG_FOLDER/NOCAPS_${TIME_START}.log"

##########################################
# MODEL CONFIGURATION (Need to be adjusted)
model_path="models.peacap_ver7_loss"
input_image_resize=680
ckpt="results/TRAIN/${SUB_FOLDER}/${EXP_NAME}/final_result_000.pt"
ext_data_path="ext_data/ext_memory_lvis.pkl"
# image_folder='data/flickr30k/images'
##########################################
# Dataset Adjustments
# path_of_val_datasets

python -u eval_evcap_bert_patch.py \
--model_path ${model_path} \
--name_of_datasets coco_val2014 \
--path_of_val_datasets ./data/coco/karpathy/captions_testKarpathy_fixed_format.json \
--image_size ${input_image_resize} \
--device cuda:$DEVICE \
--out_path=$NOCAPS_OUT_PATH \
--ckpt ${ckpt} \
--ext_data_path $ext_data_path \
--log_folder $LOG_FOLDER \
|& tee -a  ${NOCAPS_LOG_FILE}

# --model_path models.evcap_bert_patch_analysis_ver5 --image_size 224 --ckpt results/TRAIN/FULL_DATA/pecap_1_gpu_a6000/final_result_000.pt --ext_data_path ext_data/ext_memory_lvis.pkl --image_folder data/nocaps/val/images --path_of_val_datasets ./data/nocaps/nocap_val_4500_captions_fix.json
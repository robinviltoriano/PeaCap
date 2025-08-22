import sys
import os

import json
import numpy as np

import pickle

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

# os.chdir('./ext_data/vectorize_image')


import logging
import time
current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

from dataset.coco_dataset import ExtDataset

from models.evcap_modified import EVCap



def load_dataset(dataset_path = '../../data/lvis/lvis_v1_train_distilled.json'):
    with open(dataset_path) as f:
        lvis_data = json.load(f)
    
    return lvis_data

def set_dictionary_for_data_processing(dataset) -> dict:
    """
    This function sets the format for the dataset to be processed.
    The Key is the Image ID, and the Value is a list of Category IDs that belong to that image.
    
    Args:
        dataset (dict): The dataset in dictionary format.
    Returns:
        dict: A dictionary where keys are image IDs and values are lists of category IDs.
    """
    
    dict_result = dict()
    
    for item in dataset['annotations']:
        img_id = int(item['image_id'])
        cat_id = int(item['category_id'])
        
        if img_id not in dict_result:
            dict_result[img_id] = set()
            dict_result[img_id].add(cat_id)
        else:
            dict_result[img_id].add(cat_id)
            
    # Convert set to list for each image ID
    for image_id, categories in dict_result.items():
        dict_result[image_id] = list(categories)
    
    return dict_result

def get_image_file_name(img_id):
    init_id = '0'*12
    init_id_2 = init_id[:-len(str(img_id))] + str(img_id)
    file_name = f"{init_id_2}.jpg"
    return file_name

def get_dataloader(
    preprocessed_dictionary: dict,
    batch_size: int = 100
    
):
    """
    Create a DataLoader for the preprocessed dataset.
    
    Args:
        preprocessed_dictionary (dict): The dictionary containing image IDs and their corresponding category IDs.
        
    Returns:
        DataLoader: A DataLoader object for the dataset.
    """
    # Create a list of predefined image names for the Dataloader
    database_image_names = []
    for img_ids, cat_ids in preprocessed_dictionary.items():
        image_file_names = (cat_ids, get_image_file_name(img_ids))
        database_image_names.append(image_file_names)

    dataset = ExtDataset(
        database_image_names, 
        image_file_path = "./data/coco/coco2014/train2017/", 
        database_type="coco",
        input_image_size = 224,
        patchify= False,
        )
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=False, drop_last=False)
    
    return dataloader


def main(
    dataset_path: str = './lvis/lvis_v1_train_distilled.json',
    output_path_subfolder: str = './result/all_coco_dataset'
    ):
    
    # Get the dictionary for data processing
    logging.info("Setting up the dictionary for data processing")
    if not os.path.exists(os.path.join(output_path_subfolder,'preprocessing_dictionary.json')):
        # Load the dataset
        logging.info(f"Loading dataset from {dataset_path}")
        dataset = load_dataset(dataset_path)
        img_ids_for_db = set_dictionary_for_data_processing(dataset)
        
        json.dump(img_ids_for_db, open(os.path.join(output_path_subfolder,'preprocessing_dictionary.json'), 'w'), indent=4)
        logging.info(f"Preprocessing dictionary saved to {os.path.join(output_path_subfolder,'preprocessing_dictionary.json')}")
        
    else:
        logging.info(f"Preprocessing dictionary already exists at {os.path.join(output_path_subfolder,'preprocessing_dictionary.json')}, loading it.")
        img_ids_for_db = json.load(open(os.path.join(output_path_subfolder,'preprocessing_dictionary.json'), 'r'))
        
    # Create the DataLoader
    logging.info("Creating DataLoader for the dataset")
    dataloader = get_dataloader(img_ids_for_db)
    # Load the model
    logging.info("Loading the Q-Former model")
    qformer = EVCap(vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        low_resource=False,  # Set to True if you want to use low resource mode
            )
    
    qformer.eval()
    qformer = qformer.to('cuda:0')
    
    # Load Dictionary for converting category IDs to names
    id_to_catname = pickle.load(open('./data/coco/coco2014/created_usefull_file/lvis_id_to_catname.pkl', 'rb'))

    
    logging.info("Start processing the dataset")
    img_tensor_emb = []
    img_tensor_category = []
    img_tensor_id = []
    
    with torch.no_grad():
        for x in tqdm(dataloader):
            
            # move the image to GPU
            image = x['image'].to('cuda:0')
            
            # encode the image using Q-Former
            image_embeddings = qformer.just_encode_img(image)
            # image_embeddings = image_embeddings.mean(1)
            
            # store the image embeddings
            img_tensor_emb.append(image_embeddings)
            
            # store the image category
            category_list = [[int(cate.strip()) for cate in cate_batch.split(',')] for cate_batch in x['category_id']]
            id_list = [np.array(x).astype(int) for x in category_list]
            name_array = [list(np.vectorize(id_to_catname.get)(x)) for x in id_list]
            img_tensor_category.extend(name_array)
            
            # store the image id
            img_id = x['image_id']
            img_tensor_id.extend(img_id.detach().cpu().numpy())
        
    final_tensor = torch.cat(img_tensor_emb,dim=0)
    final_tensor = final_tensor.cpu().detach()
    
    # save the embeddings
    with open(os.path.join(output_path_subfolder,'ext_memory_lvis_distilled_with_img_id.pkl'), 'wb') as f:
        pickle.dump([final_tensor, img_tensor_category, img_tensor_id], f)
    
    
    
if __name__ == "__main__":
    """
    Script example:
    python -m ext_data.vectorize_image.qformer_vectorize_image
    """
    logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to terminal
        logging.FileHandler(f'./ext_data/vectorize_image/log/qformer_vectorize_image_{current_time}.log')  # Output to file
    ]
    )
    main(
        dataset_path = './data/lvis/lvis_v1_train_distilled.json',
        output_path_subfolder = './ext_data/result/embeddings_32'
    )
    
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

from save_progress_decorator import save_progress


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

def patchify(image_tensor, patch_size, patch_stride=None):
        if patch_stride is None:
            patch_stride = patch_size
        patches = image_tensor.unfold(
            1, patch_size, patch_stride).unfold(2, patch_size, patch_stride)
        patches = patches.reshape(3, -1, patch_size, patch_size).permute(1, 0, 2, 3)
        return patches 

def get_image_file_name(img_id):
    init_id = '0'*12
    init_id_2 = init_id[:-len(str(img_id))] + str(img_id)
    file_name = f"{init_id_2}.jpg"
    return file_name

def get_dataloader(
    preprocessed_dictionary: dict,
    batch_size: int = 1,
    input_image_size: int = 224,
    debug: bool = False,
    
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
        input_image_size = input_image_size,
        patchify= False,
        )
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, shuffle=False, drop_last=False)
    
    if debug:
        
        # Create a tensor of ones with shape (Num of Data, 3, 450, 450)
        ones_tensor = torch.ones(100, 3, 450, 450)
        
        class DictTensorDataset(torch.utils.data.Dataset):
            def __init__(self, tensor):
                self.tensor = tensor
            def __len__(self):
                return self.tensor.shape[0]
            def __getitem__(self, idx):
                return {'image': self.tensor[idx],
                        'category_id': '1,2,3',
                        'image_id': torch.tensor(idx+1)}

        dataset = DictTensorDataset(ones_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    return dataloader

@save_progress("ext_data/vectorize_image/log/qformer_vectorize_image_2x2_patch/result_log.pkl", save_every=250)
def forward(x, qformer, id_to_catname, is_patchify = False, image_id = None):
    # move the image to GPU
    image = x['image'].to('cuda:1')
    
    if is_patchify:
        image_patches = torch.stack([patchify(img, patch_size=224) for img in image])
        
        # Pad in case not equal to model expected input resolution.
        p = qformer.visual_encoder.image_size - 224
        image_patches_pad = torch.nn.functional.pad(
            image_patches, (p//2, p//2, p//2, p//2), "constant", 0).to('cuda:1')
        
        batch_size, num_patch, c, w, h = image_patches_pad.size()
        
        image_patches_pad = image_patches_pad.view(-1, c, w, h)
        
        image_embeddings = qformer.just_encode_img(image_patches_pad)
        
        image_embeddings = image_embeddings.view(batch_size, num_patch, 32, -1)
            
    else:
    
        # encode the image using Q-Former
        image_embeddings = qformer.just_encode_img(image)
        
        # use mean pooling to get the image embeddings
        # image_embeddings = image_embeddings.mean(1)
    
    # store the image category
    category_list = [[int(cate.strip()) for cate in cate_batch.split(',')] for cate_batch in x['category_id']]
    id_list = [np.array(x).astype(int) for x in category_list]
    name_array = [list(np.vectorize(id_to_catname.get)(x)) for x in id_list]

    
    result = {
        'image_id':image_id.detach().cpu().numpy(),
        'category': name_array,
        'embedding': image_embeddings.cpu().detach()  
    }
    
    return result

def main(
    dataset_path: str = './lvis/lvis_v1_train_distilled.json',
    output_path_subfolder: str = './result/all_coco_dataset',
    is_patchify: bool = False, 
    input_image_size: int = 224,
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
    dataloader = get_dataloader(img_ids_for_db, batch_size=10 ,input_image_size=input_image_size, debug=False)
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
    qformer = qformer.to('cuda:1')
    
    # Load Dictionary for converting category IDs to names
    id_to_catname = pickle.load(open('./data/coco/created_usefull_file/lvis_id_to_catname.pkl', 'rb'))

    
    logging.info("Start processing the dataset")
    
    with torch.no_grad():
        for x in tqdm(dataloader):
            
            image_id = x['image_id']
            forward(x = x, qformer = qformer, id_to_catname = id_to_catname, image_id=image_id, is_patchify = is_patchify)
            
    forward._final_save()
        
    saved_data = pickle.load(open('ext_data/vectorize_image/log/qformer_vectorize_image_2x2_patch/result_log.pkl', 'rb'))
    
    img_tensor_emb = []
    img_tensor_category = []
    img_tensor_id = []
    
    for x in saved_data:
        img_tensor_emb.append(x['embedding'])
        img_tensor_category.extend(x['category'])
        img_tensor_id.extend(x['image_id'])
    
    final_tensor = torch.cat(img_tensor_emb,dim=0)
    final_tensor = final_tensor.cpu().detach()
    
    # save the embeddings
    with open(os.path.join(output_path_subfolder,'ext_memory_lvis_distilled_with_img_id_2x2_patch.pkl'), 'wb') as f:
        pickle.dump([final_tensor, img_tensor_category, img_tensor_id], f)
    
    
    
if __name__ == "__main__":
    """
    Script example:
    python -m ext_data.vectorize_image.qformer_vectorize_image
    """
    import os
    
    LOG_FILE_PATH = './ext_data/vectorize_image/log/qformer_vectorize_image_2x2_patch'
    os.makedirs(LOG_FILE_PATH, exist_ok=True)
    
    logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to terminal
        logging.FileHandler(f'{LOG_FILE_PATH}/{current_time}.log')  # Output to file
    ]
    )
    main(
        dataset_path = './data/lvis/lvis_v1_train_distilled.json',
        output_path_subfolder = './ext_data/result/embeddings_32',
        is_patchify = True,
        input_image_size = 450,
    )
    
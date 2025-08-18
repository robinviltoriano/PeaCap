"""
This file is used for sampling the image from the COCO dataset.
It sample the image per category
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()] # Output to terminal    
                    )

def data_sampling(data: dict, percentage: float = 0.1) -> dict:
    """
    Sample a percentage of data from each category in the dataset. 
    This function is for getting the image id
    
    Args:
        data (dict): Key-value pairs where keys are categories and values are set of image id.
        percentage (float): The percentage of data to sample from each category.
        
    Returns:
        dict: A dictionary with the same keys as input, but with sampled values.
    """
    np.random.seed(3)
    
    sampled_data = {}
    for cat_id, image_id_list in data.items():
        # turn set to list for sampling
        image_per_category = list(image_id_list)
        
        # get the number of images to sample per category
        sampled_size = int(len(image_per_category) * percentage)
        
        # sample the images
        sampled_images = np.random.choice(image_per_category, size=sampled_size, replace=False)
        
        # store the sampled images in the dictionary
        sampled_data[cat_id] = sampled_images

    return sampled_data

def sampled_caption_file(caption_file: list, 
                         sampled_image_per_category_dict: dict,
                         karpathy_test_image_ids: list) -> list:
    """
    Sample the caption file based on the sampled image per category.
    
    Args:
        caption_file (list): The original caption file.
        sampled_image_per_category_dict (dict): The dictionary with sampled images per category.
        
    Returns:
        list: A list of sampled captions.
    """
    all_sampled_images = list(set([image for images in sampled_image_per_category_dict.values() for image in images]))

    sampled_captions = []
    for x in tqdm(caption_file):
        
        try:
            img_id = x['image_id']
        except:
            img_id = x['id']
        
        if (img_id in all_sampled_images) and (img_id not in karpathy_test_image_ids):
            sampled_captions.append(x)
            
    return sampled_captions

def plot_number_of_images_per_category(original_data, 
                                       sampled_data, 
                                       category_map_dict,
                                       output_file_name):
    """
    Plot the number of images per category before and after sampling.
    
    Args:
        original_data (dict): The original data with image ids per category.
        sampled_data (dict): The sampled data with image ids per category.
        category_map_dict (dict): The mapping of category ids to names.
    """
    original_counts = {category_map_dict[str(cat_id)]: len(images) for cat_id, images in original_data.items()}
    sampled_counts = {category_map_dict[str(cat_id)]: len(images) for cat_id, images in sampled_data.items()}

    fig, ax = plt.subplots(figsize=(20, 5))
    ax.bar(x=list(original_counts.keys()),
        height=list(original_counts.values()),
        color='skyblue', label='Number of Images')

    ax.bar(x=list(sampled_counts.keys()),
        height=list(sampled_counts.values()),
        color='lightcoral', label='Number of Sampled Images', alpha=0.7)

    ax.set_xlabel('Category ID')
    ax.set_ylabel('Number Images')
    ax.set_title('Number of Images per Category in COCO 2014 in a Logarithmic Scale')

    ax.tick_params(axis='x', labelsize=9, rotation=90)
    ax.tick_params(axis="y", length=5)
    ax.set_yscale("log")

    # remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # set y-ticks to specific values
    ax.set_yticks([10, 10**2, 10**3, 10**4, 10**5, 10**6])
    ax.set_yticklabels([10, 100, 1000, 10000, 100000, 100000])

    plt.legend()
    
    plt.savefig(output_file_name, bbox_inches='tight')
    

def main(args):
    
    logging.info(f"Sampling data from {args.instance_file} and {args.caption_file} with {args.percentage * 100}% of images per category.")
    # Load the instance file
    instance_file = json.load(open(args.instance_file, "r"))
    
    # Load the caption file
    caption_file = json.load(open(args.caption_file, "r"))
    
    # Load Category Map file
    category_map_dict = json.load(open('../data/coco/coco2014/created_usefull_file/category_map.json', "r"))
    
    # Load Karpathy Test Image IDs
    karpathy_test_image_ids = json.load(open('../data/coco/coco2014/created_usefull_file/karpathy_test_image_ids.json', "r"))
    
    logging.info(f"Creating a dictionary for image ids per category.")
    # Get Image ID per category
    image_per_category_dict = {int(cat_id): set() for cat_id in category_map_dict['map'].keys()}
    for ann in instance_file['annotations']:
        cat_id = ann['category_id']
        image_id = ann['image_id']
        
        image_per_category_dict[cat_id].add(image_id)

    logging.info("Start data sampling")
    # Get sampled image per category
    sampled_image_per_category_dict = data_sampling(image_per_category_dict, args.percentage)
    
    # Get the sampled for caption file
    logging.info("Sampling images key")
    image_key_sampled = sampled_caption_file(caption_file['images'], 
                                                sampled_image_per_category_dict, 
                                                karpathy_test_image_ids=karpathy_test_image_ids['image_ids'])
    
    logging.info("Sampling annotations key")
    annotations_key_sampled = sampled_caption_file(caption_file['annotations'], 
                                                sampled_image_per_category_dict, 
                                                karpathy_test_image_ids=karpathy_test_image_ids['image_ids'])
    
    logging.info('Saved the sampled data.')
    number_of_unique_images = len(image_key_sampled)
    number_of_captions = len(annotations_key_sampled)
    sampled_data = {
        "info": f'Sampled of {args.caption_file} with {args.percentage * 100}% of images per category. Number of unique images: {number_of_unique_images}. Number of captions: {number_of_captions}',
        'statistics':{
            "number_of_unique_images": number_of_unique_images,
            "number_of_captions": number_of_captions,
            "percentage_per_categories": args.percentage,
            "original_number_of_images": len(caption_file['images']),
            "original_number_of_captions": len(caption_file['annotations']),
            "percentage_of_images": number_of_unique_images / len(caption_file['images']),
            "percentage_of_captions": number_of_captions / len(caption_file['annotations']),
            
        },
        "images": image_key_sampled,
        "annotations": annotations_key_sampled,
        "license": caption_file['licenses'],
        
    }
    
    # Save the sampled data
    json.dump(sampled_data, open(f"../data/coco/coco2014/annotations/{args.output_file}", "w"), indent=4)
    
    logging.info('Creating a plot for the number of images per category before and after sampling.')
    # Plot the statistics
    output_file_name = args.output_file.replace('.json', '_sampling_statistics.png')
    plot_number_of_images_per_category(original_data=image_per_category_dict,
                                       sampled_data=sampled_image_per_category_dict,
                                       category_map_dict=category_map_dict['map'],
                                       output_file_name=f"result/sample_data/images/{output_file_name}")
    
    logging.info('Done!')
    
if __name__ == "__main__":
    """
    Sample arguments for the script:
    python coco_data_sampling.py --instance_file ../data/coco/coco2014/annotations/instances_train2014.json \
                                 --caption_file ../data/coco/coco2014/annotations/captions_train2014.json \
                                 --percentage 0.04 \
                                 --output_file instances_train2014_sampled.json
                                 
    --instance_file ../data/coco/coco2014/annotations/instances_val2014.json --caption_file ../data/coco/coco2014/annotations/captions_val2014.json --output_file instances_val2014_sampled.json --percentage 0.05
    """
    
    
    parser = argparse.ArgumentParser(description="Sample COCO dataset")
    parser.add_argument("--instance_file", type=str, required=True, help="Path to the instance file")
    parser.add_argument("--caption_file", type=str, required=True, help="Path to the caption file")
    parser.add_argument("--percentage", type=float, default=0.1, help="Percentage of data to sample from each category")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the sampled data")
    
    args = parser.parse_args()
    
    main(args)
    
    
    
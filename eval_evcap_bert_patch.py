import os
import json
import torch
import argparse
from tqdm import tqdm
from PIL import Image
from search import beam_search
import random
import numpy as np

import importlib
# from models.evcap_bert_patch_ver3 import EVCap
from evaluation.pycocoevalcap.eval import COCOEvalCap

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from collections import OrderedDict
from datasets import load_dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess_image(img_path, resize: int = 224) -> torch.Tensor:
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((resize, resize), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    return transform(img).unsqueeze(0)

def get_image_path(image_id: int) -> str:
    base_image_folder_path = "data/coco/coco2014/val2014/"
    dummy_id = '0'*12
    image_id = dummy_id[:-len(str(image_id))] + str(image_id)
    image_path = f"COCO_val2014_{image_id}.jpg"

    return os.path.join(base_image_folder_path, image_path)

def load_model(model_path, model_name):
    module = importlib.import_module(model_path)
    model = getattr(module, model_name)
    
    return model

def validation_whoops(
    args,
    model, 
    tokenizer,    
) -> None:

    device = args.device
    HF_TOKEN = os.getenv("HF_TOKEN")

    predicts = []
    examples = load_dataset('nlphuji/whoops', token=HF_TOKEN)
    model.eval()
    for example in examples['test']:
        image_id = example['image_id']
        captions = example['crowd_captions']
        print('\n')
        print(image_id)
        print('GT: ', captions)
        image = example['image']
        image = preprocess_image(image).to(device)
        with torch.cuda.amp.autocast(enabled=True):
            qform_all_proj, atts_qform_all_proj  = model.encode_img(image)
            prompt_embeds, atts_prompt = model.prompt_wrap(qform_all_proj, atts_qform_all_proj, model.prompt_list)
            tokenizer.padding_side = "right"
            batch_size = qform_all_proj.shape[0]
            bos = torch.ones([batch_size, 1],
                            device=image.device) * tokenizer.bos_token_id
            bos = bos.long()
            bos_embeds = model.llama_model.model.embed_tokens(bos)
            embeddings = torch.cat([bos_embeds, prompt_embeds], dim=1)
            sentence = beam_search(embeddings = embeddings, tokenizer = tokenizer, beam_width = args.beam_width, model = model.llama_model)
            sentence = sentence[0]
            print('Pred: ', sentence)
  
        predict = {}
        predict["split"] = 'valid'
        predict["image_name"] = image_id
        predict["captions"] = captions
        predict["prediction"] = sentence
        predicts.append(predict)
    
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)
    out_json_path = os.path.join(args.out_path, f'{args.name_of_datasets}_generated_captions.json')
    with open(out_json_path, 'w') as outfile:
        json.dump(predicts, outfile, indent = 4)




def validation_coco_flickr30k(
    args,
    inpath, 
    model,
    tokenizer, 
) -> None:

    device = args.device
    with open(inpath, 'r') as infile:
        annotations = json.load(infile)
    predicts = []
    for idx, item in tqdm(enumerate(annotations)):
        image_id = item
        captions = annotations[item]
        image_path = args.image_folder + image_id
        print('\n')
        print(image_path)
        print('GT: ', captions)
        image = preprocess_image(image_path).to(device)
        with torch.cuda.amp.autocast(enabled=True):
            qform_all_proj, atts_qform_all_proj  = model.encode_img(image)
            prompt_embeds, atts_prompt = model.prompt_wrap(qform_all_proj, atts_qform_all_proj, model.prompt_list) #(self, img_embeds, batch_names, atts_img, prompt_list):
            tokenizer.padding_side = "right"
            batch_size = qform_all_proj.shape[0]
            bos = torch.ones([batch_size, 1],
                            device=image.device) * tokenizer.bos_token_id
            bos = bos.long()
            bos_embeds = model.llama_model.model.embed_tokens(bos)
            embeddings = torch.cat([bos_embeds, prompt_embeds], dim=1)
            sentence = beam_search(embeddings = embeddings, tokenizer = tokenizer, beam_width = args.beam_width, model = model.llama_model) # List[str]
            sentence = sentence[0]
            print('Pred: ', sentence)
  
        predict = {}
        predict["split"] = 'valid'
        predict["image_name"] = image_id
        predict["captions"] = captions
        predict["prediction"] = sentence
        predicts.append(predict)
    
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)
    out_json_path = os.path.join(args.out_path, f'{args.name_of_datasets}_generated_captions.json')
    with open(out_json_path, 'w') as outfile:
        json.dump(predicts, outfile, indent = 4)
        
def validation_coco_val2014(
    args,
    inpath, 
    model,
    tokenizer,
    ) -> None:
    
    device = args.device
    with open(inpath, 'r') as file:
        annotations = json.load(file)

    predicts = []
    for idx, item in tqdm(enumerate(annotations['annotations'])):
        image_id = item['image_id']
        captions = item['caption']
        image_path = get_image_path(image_id)
        print('\n')
        print(image_path)
        print('GT: ', captions)
        image = preprocess_image(image_path, resize = args.image_size).to(device)
        with torch.autocast('cuda', enabled=True):
            qform_all_proj, atts_qform_all_proj  = model.encode_img(image)
            prompt_embeds, _ = model.prompt_wrap(qform_all_proj, atts_qform_all_proj, model.prompt_list) #(self, img_embeds, batch_names, atts_img, prompt_list):
            tokenizer.padding_side = "right"
            batch_size = qform_all_proj.shape[0]
            bos = torch.ones([batch_size, 1],
                            device=image.device) * tokenizer.bos_token_id
            bos = bos.long()
            bos_embeds = model.llama_model.model.embed_tokens(bos)
            embeddings = torch.cat([bos_embeds, prompt_embeds], dim=1)
            sentence = beam_search(embeddings = embeddings, tokenizer = tokenizer, beam_width = args.beam_width, model = model.llama_model) # List[str]
            sentence = sentence[0]
            print('Pred: ', sentence)
  
        predict = {}
        predict["split"] = 'valid'
        predict["image_name"] = image_id
        predict["captions"] = captions
        predict["prediction"] = sentence
        predicts.append(predict)
        
        if not os.path.exists(args.out_path):
            os.makedirs(args.out_path, exist_ok=True)
        out_json_path = os.path.join(args.out_path, f'{args.name_of_datasets}_generated_captions.json')
        with open(out_json_path, 'w') as outfile:
            json.dump(predicts, outfile, indent = 4)

def validation_nocaps(
    args,
    inpath,
    model,        
    tokenizer,            
) -> None:
    device = args.device
    with open(inpath, 'r') as infile:
        annotations = json.load(infile)
    indomain = []
    neardomain = []
    outdomain = []
    overall = []
    img_info = json.load(open('/home/nlab/li/research/3_NOC/ours_blip/M_MiniGPT-4/data/nocaps/nocaps_val.json','r'))
    model.eval()
    for idx, annotation in tqdm(enumerate(annotations)):
        ann = img_info[idx]
        image_file = ann['image']
        img_id = ann['img_id']
        image_id = annotation['image_id']
        split = annotation['split']
        captions = annotation['caption']
        print('\n')
        image_path = args.image_folder + '/' + image_file
        print(image_path)
        print('GT: ', captions)
        image = preprocess_image(image_path).to(device)
        with torch.autocast('cuda', enabled=True):
            qform_all_proj, atts_qform_all_proj  = model.encode_img(image)
            prompt_embeds, atts_prompt = model.prompt_wrap(qform_all_proj, atts_qform_all_proj, model.prompt_list) #(self, img_embeds, batch_names, atts_img, prompt_list):
            tokenizer.padding_side = "right"
            batch_size = qform_all_proj.shape[0]
            bos = torch.ones([batch_size, 1],
                            device=image.device) * tokenizer.bos_token_id
            bos = bos.long()
            bos_embeds = model.llama_model.model.embed_tokens(bos)

            embeddings = torch.cat([bos_embeds, prompt_embeds], dim=1)

            sentence_ = beam_search(embeddings = embeddings, tokenizer = tokenizer, beam_width = args.beam_width, model = model.llama_model)
            sentence_ = sentence_[0]
            sentence = sentence_.split('#')[0]
            print('Pred: ', sentence)

            predict = {}
            predict["split"] = split
            predict["image_name"] = image_id
            predict["captions"] = captions
            predict["prediction"] = sentence

            overall.append(predict)
            if split == 'in_domain':
                indomain.append(predict)
            elif split == 'near_domain':
                neardomain.append(predict)
            elif split == 'out_domain':
                outdomain.append(predict)

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)

    with open(os.path.join(args.out_path, f'overall_generated_captions.json'), 'w') as outfile:
        json.dump(overall, outfile, indent = 4)
    with open(os.path.join(args.out_path, f'indomain_generated_captions.json'), 'w') as outfile:
        json.dump(indomain, outfile, indent = 4)
    with open(os.path.join(args.out_path, f'neardomain_generated_captions.json'), 'w') as outfile:
        json.dump(neardomain, outfile, indent = 4)
    with open(os.path.join(args.out_path, f'outdomain_generated_captions.json'), 'w') as outfile:
        json.dump(outdomain, outfile, indent = 4)
        
def get_score(result_file):
    # Manually set JAVA_HOME for Python
    os.environ['JAVA_HOME'] = '/home/robin/java/jdk-24'
    os.environ['PATH'] = os.environ['JAVA_HOME'] + '/bin:' + os.environ['PATH']
    
    cocoEval = COCOEvalCap(result_file)
    cocoEval.evaluate()


@torch.no_grad()
def main(args, model_config) -> None:
    # initializing
    device = args.device
    # loading model
    # ckpt = 'results/train_evcap/000.pt' ## change
    ckpt = args.ckpt
    print('load:', ckpt)
    
    EVCap = load_model(args.model_path, args.model_name)

    model = EVCap(**model_config)
    state_dict = torch.load(ckpt, map_location=device)['model']


    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    inpath = args.path_of_val_datasets
    tokenizer = model.llama_tokenizer
    if args.name_of_datasets == "nocaps":
        validation_nocaps(args, inpath, model, tokenizer)
    if args.name_of_datasets == "coco" or args.name_of_datasets == "flickr30k":
        validation_coco_flickr30k(args, inpath, model, tokenizer)
    if args.name_of_datasets == "whoops":
        validation_whoops(args, model, tokenizer)
    if args.name_of_datasets == "coco_val2014":
        validation_coco_val2014(args, inpath, model, tokenizer)

    print('Getting score ...')
    get_score(f"{args.out_path}/coco_val2014_generated_captions.json")
    print('Done!')
    
if __name__ == '__main__':
    print('Starts ...')
    print(" # PID :", os.getpid())
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str, required=True, help = 'path to the model')
    parser.add_argument('--device', default = 'cuda:0')
    parser.add_argument('--model_name', type = str, default = 'EVCap', help = 'name of the model class')
    parser.add_argument('--name_of_datasets', default = 'coco_val2014', choices = ('coco', 'flickr30k', 'nocaps', 'whoops','coco_val2014'))
    parser.add_argument('--image_size', type = int, default = 224, help = 'image size for preprocessing')
    parser.add_argument('--path_of_val_datasets', default = 'data/coco/karpathy/caption_testKarpathy_in_scope_val_reference.json')
    parser.add_argument('--image_folder', default = 'data/coco/coco2014/val2014/')
    parser.add_argument('--out_path', default = './generated_captions.json')
    parser.add_argument('--num_query_token_txt', type = int, default = 8)
    parser.add_argument('--topn', type = int, default = 9)
    parser.add_argument('--beam_width', type = int, default = 5, help = 'width of beam')
    parser.add_argument('--random_seed', type = int, default = 42, help = 'set random seed for reproducing')
    parser.add_argument('--ext_data_path', type = str, default='ext_data/ext_memory_original_sample.pkl', help = 'path to the external data')
    parser.add_argument('--ckpt', type = str, default='results/train_evcap_bert_patch_ver4_in_scope/final_result_000.pt', help = 'path to the checkpoint')
    parser.add_argument('--log_folder', type = str, default = 'logs', help = 'folder to save logs')
    args = parser.parse_args()
    set_seed(args.random_seed)
    print('args: {}\n'.format(vars(args)))
    
    model_config = {
        "ext_path" : args.ext_data_path,
        "vit_model" : "eva_clip_g",
        "q_former_model" : "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        "patch_size" : 224,
        # "img_size": 224, ##
        "drop_path_rate" : 0,
        "use_grad_checkpoint" : False,
        "vit_precision" : "fp16",
        "freeze_vit" : True,
        "freeze_qformer" : True,
        "num_query_token" : 32,
        "topn" :  args.topn,
        "llama_model" : "lmsys/vicuna-7b-v1.3",
        "prompt_path" : "prompts/prompt_evcap.txt",
        "prompt_template" : '###Human: {} ###Assistant: ',
        "max_txt_len" : 128,
        "end_sym" : '\n',
        "low_resource" : False,
        "device_8bit" : 0,
    }
    
    # Log Model Configuration and Arguments
    log_data = {
        'model_config' : model_config,
        'args' : vars(args)
    }
    
    log_folder = f'{args.log_folder}'
    print(f"Saving model configuration and arguments to {os.path.join(log_folder, 'model_config_and_arguments.json')}")
    json.dump(log_data, open(os.path.join(log_folder, 'model_config_and_arguments.json'), 'w'), indent=4)
    
    main(args, model_config=model_config)
    
import os
import re
from PIL import Image
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset


class COCODataset(Dataset):

    def __len__(self) -> int:
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        img_file = f'COCO_train2014_{int(ann["image_id"]):012d}.jpg'
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        caption = ann["caption"]
        selected_patch = ann.get('selected_patch', "")  # Default to -999 if not present
        
        if isinstance(selected_patch, list):
            # We transform the dynamic list of selected patches into a string
            # This is done since DataLoader does not support dynamic list sizes
            selected_patch = ', '.join(map(str, selected_patch))
            
        return {
            "image": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
            "selected_patch": selected_patch
        }

    def __init__(self, data_root, annotation_file, resize=224):
        ann_path = os.path.join(data_root, annotation_file)
        self.vis_root=os.path.join(data_root, 'train2014')
        self.annotation = []
        self.annotation.extend(json.load(open(ann_path, "r"))['annotations'])
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
        self.transform = transforms.Compose([
            transforms.Resize((resize, resize), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
class COCODataset2(Dataset):

    def __len__(self) -> int:
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        img_file = f'COCO_train2014_{int(ann["image_id"]):012d}.jpg'
        image_path = os.path.join(self.vis_root, img_file)
        image = self.clip_processor(Image.open(image_path).convert("RGB"))
        caption = ann["caption"]
        return {
            "image": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }

    def __init__(self, data_root, annotation_file, clip_processor):
        ann_path = os.path.join(data_root, annotation_file)
        self.vis_root=os.path.join(data_root, 'train2014')
        self.annotation = []
        self.annotation.extend(json.load(open(ann_path, "r"))['annotations'])
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
                
        self.clip_processor = clip_processor
        
class COCODatasetSelectedPatch(Dataset):
    
    def patchify_tensor(self, img_tensor, patch_stride=None):
            if patch_stride is None:
                patch_stride = self.patch_size
            patches = img_tensor.unfold(
                1, self.patch_size, patch_stride).unfold(2, self.patch_size, patch_stride)
            patches = patches.reshape(3, -1, self.patch_size, self.patch_size).permute(1, 0, 2, 3)
            return patches

    def __len__(self) -> int:
        return len(self.annotation)

    def __getitem__(self, index: int):
        ann = self.annotation[index]
        img_id = ann["image_id"]
        img_file = f'COCO_train2014_{int(img_id):012d}.jpg'
        image_path = os.path.join(self.vis_root, img_file)
        
        selected_patch = ann.get('selected_patch', None)
            
        image = Image.open(image_path).convert("RGB")
        
        image = self.transform(image)
        image = image.to(torch.float32)
        caption = ann["caption"]
        
        if self.patchify:
            image = self.patchify_tensor(image, patch_stride=self.patch_stride)
            
            if selected_patch is not None:
                selected_idx = torch.tensor(selected_patch)  # example
                mask = torch.zeros_like(image)
                mask[selected_idx] = image[selected_idx]
                image = mask
                
            if self.merge_patch:
                image = image.view(self.number_of_patch, self.number_of_patch, 3, 224, 224)
                image = image.permute(2, 0, 3, 1, 4)  
                image = image.contiguous().view(3, self.number_of_patch*224, self.number_of_patch*224)
                
                image = F.interpolate(image.unsqueeze(0), size=(self.patch_size, self.patch_size), mode='bicubic', align_corners=False).squeeze(0)
        
        return {
            "image": image,
            "text_input": caption,    
            "image_id": img_id 
        }

    def __init__(self, 
                 data_root: str, 
                 annotation_file: list, 
                 resize: int = 224,
                 is_normalize: bool = True,
                 patchify: bool = False,
                 patch_size: int = 224,
                 patch_stride: int = None,
                 merge_patch: bool = False
                 ):
        
        ann_path = os.path.join(data_root, annotation_file)
        self.vis_root=os.path.join(data_root, 'train2014')
        self.annotation = []
        self.annotation.extend(json.load(open(ann_path, "r"))['annotations'])
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

        self.is_normalize = is_normalize
        self.patchify = patchify
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.merge_patch = merge_patch
        
        if patchify:
            self.number_of_patch = resize//patch_size
        
        self.transform = transforms.Compose([
                transforms.Resize((resize, resize), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])
        
        if is_normalize:
            self.transform = transforms.Compose([
                transforms.Resize((resize, resize), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
            ])

class ExtDataset(Dataset):
    
    def patchify_tensor(self, img_tensor, patch_stride=None):
        if patch_stride is None:
            patch_stride = self.patch_size
        patches = img_tensor.unfold(
            1, self.patch_size, patch_stride).unfold(2, self.patch_size, patch_stride)
        patches = patches.reshape(3, -1, self.patch_size, self.patch_size).permute(1, 0, 2, 3)
        return patches

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        img_file = self.files[index]
        
        if self.database_type == 'lvis':
            # for LVIS database
            _, img_id, cat_id, _ = re.split(r'[_\.]', self.files[index])
            
        elif self.database_type == 'coco':
            # for COCO database
            if isinstance(self.files[index], str):
                cat_id, img_file = self.files[index].split('_')
            elif isinstance(self.files[index], tuple):
                cat_id, img_file = self.files[index]
                
                cat_id = ", ".join([str(x) for x in cat_id])
            img_id = int(re.search('\d*', img_file)[0])
            
        elif self.database_type == 'synthetic_image':
            img_id = self.files[index].split('.')[0]
            cat_id = ' '.join(re.findall(r'[a-zA-Z]+', img_id))
        else:
            # for predefined database from EVCap
            cat_id = -999
            img_id = int(re.findall('\d*',self.files[index].split('/')[1])[0])
            
        selected_patch = self.selected_patch_data.get(img_id) if self.selected_patch_data is not None else None
            
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")
        
        if self.is_transform:
            image = self.transform(image)
            image = image.to(torch.float32)
        
        if self.patchify:
            image = self.patchify_tensor(image, patch_stride=self.patch_stride)
            
            if selected_patch is not None:
                selected_idx = torch.tensor(selected_patch['selected_patch'])  # example
                mask = torch.zeros_like(image)
                mask[selected_idx] = image[selected_idx]
                image = mask
                
            if self.merge_patch:
                image = image.view(self.number_of_patch, self.number_of_patch, 3, 224, 224)
                image = image.permute(2, 0, 3, 1, 4)  
                image = image.contiguous().view(3, self.number_of_patch*224, self.number_of_patch*224)
                
                image = F.interpolate(image.unsqueeze(0), size=(self.patch_size, self.patch_size), mode='bicubic', align_corners=False).squeeze(0)
        
        return {
            "image": image,
            "image_id": img_id,
            "category_id": cat_id,            
        }

    def __init__(self, 
                 image_files: list, 
                 image_file_path: str = './lvis/', 
                 database_type: str = 'lvis',
                 input_image_size: int = 224,
                 is_normalize: bool = True,
                 patchify: bool = False,
                 patch_size: int = 224,
                 patch_stride: int = None,
                 selected_patch_data: dict = None,
                 merge_patch: bool = False,
                 is_transform: bool = True
                 ):
        self.files = image_files  
        self.vis_root = image_file_path
        self.is_normalize = is_normalize
        self.patchify = patchify
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.selected_patch_data = selected_patch_data
        self.merge_patch = merge_patch
        self.is_transform = is_transform
        
        if patchify:
            self.number_of_patch = input_image_size//patch_size
        
        self.transform = transforms.Compose([
                transforms.Resize((input_image_size, input_image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])
        
        if is_normalize:
            self.transform = transforms.Compose([
                transforms.Resize((input_image_size, input_image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
            ])
        self.database_type = database_type 
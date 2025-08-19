import os
import json
from collections import OrderedDict
from tqdm import tqdm
import importlib
import torch
from torch.utils.data import DataLoader, DistributedSampler
import random
import sys
import argparse
import numpy as np
import utils
from optims import LinearWarmupCosineLRScheduler, set_optimizer

from dataset.coco_dataset import COCODataset
from common.dist_utils import (
    get_rank,
    init_distributed_mode,
    get_world_size,
)

from plot_training_metrics import plot_training_metrics
import torch.distributed as dist
import glob

def load_model(model_path, model_name):
    module = importlib.import_module(model_path)
    model = getattr(module, model_name)
    
    return model

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(model, optimizer, scaler, loss, cur_epoch, cur_step, output_dir):
    """
    Save the checkpoint at the current epoch.
    """
    # full_state_dict = model.state_dict()
    model_no_ddp = model
    param_grad_dic = {
        k: v.requires_grad for (k, v) in model_no_ddp.named_parameters()
    }
    state_dict = model_no_ddp.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            del state_dict[k]
        else:
            state_dict[k] = state_dict[k].cpu()
            
    save_obj = {
        "model": state_dict,
        "optimizer": {k: v.cpu() if torch.is_tensor(v) else v for k, v in optimizer.state_dict().items()},
        "scaler": scaler.state_dict(),
        "loss": loss,
        "epoch": cur_epoch,
        "step": cur_step,
    }
    print("Saving checkpoint at step {} to {}.".format(cur_step, output_dir))
    torch.save(save_obj, output_dir)


def train(dataset, model, args):
    if 'cuda' in args.device:
        device = torch.device(f"cuda:{get_rank()}")
        # device = torch.device(args.device)
    else:
        device = 'cpu'
        
    batch_size = args.bs
    epochs = args.epochs
    accum_grad_iters = args.accum_grad_iters
    output_dir = args.out_dir
    
    # curr_loss = torch.inf
    # best_loss = torch.inf
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    if args.distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=True,
            num_replicas=get_world_size(),
            rank=get_rank(),
            seed=3,
        )
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[get_rank()]) #, find_unused_parameters=True
    else: 
        sampler = None
        model = model.to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, sampler=sampler, shuffle=False, drop_last=True)
    model.train()
    init_lr = 1e-4 
    optimizer = set_optimizer(model, init_lr=init_lr, weight_decay=0.05)
    scheduler = LinearWarmupCosineLRScheduler(optimizer= optimizer,
                max_epoch=epochs,
                iters_per_epoch=len(dataloader),
                min_lr=1e-5 ,
                init_lr=init_lr,
                decay_rate=None,
                warmup_start_lr=1e-6 ,
                warmup_steps=int(0.05 * (len(dataset) // args.accum_grad_iters)),)
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()
    use_amp = scaler is not None
    print('use_amp', use_amp)
    
    # Metric Logger Configuration
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=args.accum_grad_iters, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=args.accum_grad_iters, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=args.accum_grad_iters, fmt='{value:.6f}'))
    
    if args.ckpts_path:
        print(f"Loading checkpoint from {args.ckpts_path}")
        ckpts = torch.load(args.ckpts_path, map_location=device)
        
        cur_epoch = ckpts['epoch']
        cur_step = ckpts['step']
        
        # Load pretrained model weights
        new_state_dict = OrderedDict()
        for k, v in ckpts['model'].items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
            
        model.load_state_dict(new_state_dict, strict=False)
        model.to(device)
        
        # Load optimizer state
        optimizer.load_state_dict(ckpts['optimizer'])
        
        # Set scheduler and Data Loader to the current step
        print("Resuming training from epoch {}, step {}".format(cur_epoch, cur_step))
        train_iter = iter(dataloader)
        for i in tqdm(range(cur_step)):
            scheduler.step(cur_epoch=cur_epoch, cur_step=i)
            next(train_iter)
            
        train_dataloader = train_iter

        # Set sampler epoch
        if args.distributed:
            sampler.set_epoch(cur_epoch)
        
        scaler.load_state_dict(ckpts['scaler'])
        
        metric_logger.update(loss=ckpts['loss'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=0.0)  # No grad norm available in checkpoint
        
        del dataloader
        del ckpts
    
    else:
        train_dataloader = dataloader
        cur_step = 0
        
        metric_logger.update(loss=1000.0)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=0.0)  # No grad norm available at the start
        
        del dataloader
            
    for epoch in range(epochs):
        print(f">>> Training epoch {epoch}")
        sys.stdout.flush()
        
        print_freq = 50
        header = 'Train Epoch: [{}]'.format(epoch)
        for idx, samples in enumerate(metric_logger.log_every(train_dataloader, print_freq, header, cur_step=cur_step), start=cur_step):
            
            # try:
            samples['image'] = samples['image'].to(device)
            
            if cur_step != 0:
                pass
            else:
                scheduler.step(cur_epoch=epoch, cur_step=idx)    
                
            with torch.autocast('cuda', enabled=use_amp):
                loss = model(samples)["loss"]
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            # Gradient Clipping
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e6)
                
            # Calculate Gradients
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            if (idx + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()                     
                else:    
                    optimizer.step()
                optimizer.zero_grad()
            metric_logger.update(loss=loss.detach().cpu().item())
            metric_logger.update(lr = optimizer.param_groups[0]["lr"])
            metric_logger.update(grad_norm=total_norm)
            
            # cur_loss = loss.detach().cpu().item()
            # if cur_loss < best_loss:
            #     best_loss = cur_loss
            #     print(f"New best loss: {best_loss:.6f} at epoch {epoch}, step {idx}")
            #     save_checkpoint(model, optimizer, scaler, loss, epoch, idx, os.path.join(output_dir, f"best_loss_model.pt"))
            #     torch.cuda.empty_cache()
            
            if idx % 5000 == 0 and idx > 0:
                output_dir_model = os.path.join(output_dir, f"{idx}.pt")
                save_checkpoint(model, optimizer, scaler, loss, epoch, idx, output_dir_model)
                torch.cuda.empty_cache()
                    
            # except Exception as e:
            #     print(f"Error at epoch {epoch}, step {idx}: {e}")
            #     continue
    
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger.global_avg())
 
        if epoch == epochs - 1:
            
            try:
                # Delete all checkpoints except the last one
                for file in glob.glob(f"{output_dir}/*.pt"):
                    os.remove(file)
            except:
                pass
            # Save the final model checkpoint
            output_dir_model = os.path.join(output_dir, f"final_result_{epoch:03d}.pt")
            save_checkpoint(model, optimizer, scaler, loss, epoch, idx, output_dir_model)

            
            torch.cuda.empty_cache()
            
        metric_logger.save_loss_lr(os.path.join(output_dir, "training_metrics.json"))
    return model


def main():
    # os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:128"
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print('Starts ...')
    print(" # PID :", os.getpid())
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model module')
    parser.add_argument('--model_name', type=str, default='EVCap')
    parser.add_argument('--out_dir', default='./checkpoints')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--is_rn', dest='is_rn', action='store_true')
    parser.add_argument('--device', default = 'cuda', help = 'gpu for training') ## 
    parser.add_argument('--distributed', default = True)
    parser.add_argument('--amp', default = True)
    parser.add_argument('--dist_url', default = "env://")
    parser.add_argument('--world_size', type = int, default = 1)
    parser.add_argument('--num_query_token_txt', type = int, default = 8)
    parser.add_argument('--topn', type = int, default = 9)
    parser.add_argument('--disable_random_seed', action = 'store_true', default = False, help = 'set random seed for reproducing')
    parser.add_argument('--random_seed', type = int, default = 42, help = 'set random seed for reproducing')
    parser.add_argument('--annotation_file_for_train', type = str, default = 'annotations/captions_train2014_10_categories.json')
    parser.add_argument('--low_resource', type = bool, default = False)
    parser.add_argument('--ckpts_path', type = str, default = None, help = 'the path of the checkpoints to load')
    parser.add_argument('--log_folder', type = str, default = 'logs', help = 'folder to save logs')
    parser.add_argument('--ext_path', type = str, default = 'ext_data/sample_10_categories/ext_memory_original_format.pkl', help = 'the path of the external memory')
    parser.add_argument('--input_image_resize', type = int, default = 680, help = 'the input image resize size')
    parser.add_argument('--accum_grad_iters', type=int, default=1, help='Number of iterations to accumulate gradients before updating the model')
    args = parser.parse_args()
    print(f'args: {vars(args)}')
    if not args.disable_random_seed:
        set_seed(args.random_seed)
    init_distributed_mode(args)
    print(f'args: {vars(args)}')
    data_root = 'data/coco/coco2014'
    dataset = COCODataset(data_root=data_root, annotation_file=args.annotation_file_for_train, resize=args.input_image_resize)
    model_type = "lmsys/vicuna-7b-v1.3"
    
    model_config = {
        "ext_path": args.ext_path,
        "vit_model":"eva_clip_g",
        "q_former_model":"https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        "patch_size":224,
        "drop_path_rate":0,
        "use_grad_checkpoint":False,
        "vit_precision":"fp16",
        "freeze_vit":True,
        "freeze_qformer":True,
        "num_query_token":32,
        # "num_query_token_txt":8,
        "topn": args.topn,
        "llama_model":model_type,
        "prompt_path":"prompts/prompt_evcap.txt",
        "prompt_template":'###Human: {} ###Assistant: ',
        "max_txt_len":128,
        "end_sym":'\n',
        "low_resource":False,  # use 8 bit and put vit in cpu ##
        "device_8bit":0,  # the device of 8bit model should be set when loading and cannot be changed anymore.
    }
    
    # Log Model Configuration and Arguments
    log_data = {
        'model_config' : model_config,
        'args' : vars(args)
    }
    log_folder = f'{args.log_folder}'
    print(f"Saving model configuration and arguments to {os.path.join(log_folder, 'model_config_and_arguments.json')}")
    json.dump(log_data, open(os.path.join(log_folder, 'model_config_and_arguments.json'), 'w'), indent=4)
    
    EVCap = load_model(args.model_path, args.model_name)
    model = EVCap(**model_config)
    
    train(dataset, model, args)
    
    # Save training metrics in a JPG file
    plot_training_metrics(args.out_dir, args.model_path,'training_metrics.json')
    
    # Clean up
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

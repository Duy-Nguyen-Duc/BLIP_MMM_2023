import os
from tqdm import tqdm
import gc
import math
import torch
import yaml
import copy
import time
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from .data_prepare import mmm_rt_train, mmm_rt_retrieval_eval, coco_karpathy_train, coco_karpathy_retrieval_eval
from models.blip import blip_retrieval
from utils.pos_embed import interpolate_pos_embed
from collections import defaultdict
import numpy as np
from .evaluate import evaluation
from utils.metric_logger import MetricLogger, SmoothedValue

config = yaml.safe_load("config.yaml")

def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
#Create dataset
def create_dataset(dataset, config, min_scale=0.8):

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(config['image_size'],scale=(min_scale, 1.0),interpolation=InterpolationMode.BICUBIC),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAdjustSharpness(sharpness_factor=2.0),
            transforms.RandomApply([
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10)
            ], p=0.8),

            transforms.ToTensor(),
            normalize,
        ])
    transform_test = transforms.Compose([
        transforms.Resize((config['image_size'],config['image_size']),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])
    if dataset =="retrieval_coco":
        train_dataset = coco_karpathy_train(transform_train, config['image_root'], config['ann_root'])
        val_dataset = coco_karpathy_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'val')
        test_dataset = coco_karpathy_retrieval_eval(transform_test, config['image_root'], config['ann_root'], 'test')          


    
        return train_dataset, val_dataset, test_dataset
    elif dataset =="retrieval_mmm_rt":
        train_dataset = mmm_rt_train(transform_train, config['image_root'], config['ann_root'], config["filename_train"])
        val_dataset = mmm_rt_retrieval_eval(transform_test, config['image_root'], config['ann_root'], config["filename_val"])
        
        return train_dataset, val_dataset
    
    
    return train_dataset


#Create sampler and dataloader 
def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
            
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders

#### Dataset #### 
print("Creating retrieval dataset")
train_dataset, val_dataset = create_dataset('retrieval_%s'%config['dataset'], config)
samplers = [None, None, None]
train_loader, val_loader = create_loader([train_dataset, val_dataset],samplers,
                            batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                            num_workers=[4,4,4],
                            is_trains=[True, False, False],
                            collate_fns=[None,None,None])
#### Model ####
print("Creating model")
model = blip_retrieval(med_config = config["med_config"],pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'],
                        vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'],
                        queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'])
model = model.to(config["device"])
model_without_ddp = model
optimizer = torch.optim.AdamW(params=model.parameters(), lr=float(config['init_lr']), weight_decay=config['weight_decay'])

@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):
    
    #Images->Text 
    ranks = np.zeros(scores_i2t.shape[0])
    for index,score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
  
    #Text->Images 
    ranks = np.zeros(scores_t2i.shape[0])
    
    for index,score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)        

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result =  {'txt_r1': tr1,
                    'txt_r5': tr5,
                    'txt_r10': tr10,
                    'txt_r_mean': tr_mean,
                    'img_r1': ir1,
                    'img_r5': ir5,
                    'img_r10': ir10,
                    'img_r_mean': ir_mean,
                    'r_mean': r_mean}
    return eval_result

def train_one_epoch(model, data_loader, optimizer, epoch, device):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_itm', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss_ita', SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    
    dataset_size = 0
    running_loss = 0.0
    epoch_loss = 0.0
    print("device:",device)
    bar = tqdm(enumerate(data_loader), total=len(data_loader))

    for i, (image, caption, idx) in bar:
        image = image.to(device, non_blocking=True)
        batch_size = image[0]
        idx = idx.to(device, non_blocking= True)
        if epoch>0:
            alpha = config["alpha"]
        else:
            alpha = config["alpha"]*min(1, i/len(train_loader))
        loss_ita, loss_itm = model(image, caption, alpha=alpha, idx=idx)
        loss = loss_ita + loss_itm
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_ita=loss_ita.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        bar.set_postfix(Epoch=epoch, Train_Loss=loss_ita,
                        LR=optimizer.param_groups[0]['lr'])
        
    print("Averaged stats:", metric_logger.global_avg())
    print("return value:",{k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()})
    gc.collect()
    return {k: "{:.3f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

import wandb

try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("wandb_api")
    wandb.login(key=api_key)
    anony = None
except:
    anony = "must"
    print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')
def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args["distributed"] = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}, word {}): {}'.format(
        args.rank, args.world_size, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0) 
    
init_distributed_mode(config) 
config["device"]

run = wandb.init(project="blip testing", 
                 config=config,
                 job_type='Train',
                 tags=["blip mmm retrieval"],
                 name="BLIP-mmm-baseline",
                 anonymous='must')

train_loader, val_dataset = create_loader([train_dataset, val_dataset],samplers,
                            batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                            num_workers=[4,4,4],
                            is_trains=[True, False, False],
                            collate_fns=[None,None,None])


wandb.watch(model, log_freq=100)

start = time.time()
best_model_wts = copy.deepcopy(model.state_dict())
best_epoch_loss = 0
history = defaultdict(list)

for epoch in range(0, config['max_epoch']):
    cosine_lr_schedule(optimizer, epoch, config['max_epoch'], float(config['init_lr']), float(config['min_lr']))

    train_epoch_loss = train_one_epoch(model, train_loader, optimizer, epoch, config["device"])

    #score_val_i2t, score_val_t2i, = evaluation(model_without_ddp, val_loader, device, config)
    #score_test_i2t, score_test_t2i = evaluation(model_without_ddp, test_loader, device, config)
    print(config)
    score_val_i2t, score_val_t2i, = evaluation(model_without_ddp, val_loader, config["device"], config)

    history['Train ITM Loss'].append(train_epoch_loss["loss_itm"])
    history['Valid Image to Text Loss'].append(score_val_i2t)
    history['Valid Text to Image Loss'].append(score_val_t2i)

    # Log the metrics
    wandb.log({"Learning rate": train_epoch_loss["lr"]})
    wandb.log({"Train ITM Loss": train_epoch_loss["loss_itm"]})
    wandb.log({"Train ITA Loss": train_epoch_loss["loss_ita"]})
    wandb.log({"Valid Image to Text Loss":score_val_i2t})
    wandb.log({"Valid Text to Image Loss":score_val_t2i})

    val_result = itm_eval(score_val_i2t, score_val_t2i, val_loader.dataset.txt2img, val_loader.dataset.img2txt)  
    print("val_result:",val_result)
    if epoch%1==0:
        save_path = os.path.join(config['output_dir'], 'checkpoint_epoch_'+str(epoch)+'.pth')
        print(f"Object save at:{save_path}")
        save_obj = {
            'epoch': epoch,
            'model': model.state_dict(),
            'config': config,
            'epoch_loss': val_result["r_mean"],
            'optimizer' : optimizer.state_dict(),
        }
        torch.save(save_obj, save_path)
        

    torch.cuda.empty_cache()

end = time.time()
time_elapsed = end - start
print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
       time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
print("Best Loss: {:.4f}".format(best_epoch_loss))

run.finish()
print()
wandb.finish()

# import core packages
import os
import math
import argparse
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchinfo import summary
from collections import defaultdict
from tqdm import tqdm
import wandb
# import local packages
from configs import parse_config
from data import get_transformation, get_train_dataset, get_val_dataset
from models import CL2P
from utils import get_optimizer, parse_args
from utils import reproducibility_setting, get_device, save_checkpoint, load_checkpoint
from test import extract_features, test_clustering, test_classification
from utils.visualization import sampling, reconstruction, pseudo_sampling


# get process rank, local rank and total rank number 
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))



wandb.login(key="") # login wandb


def init_distributed_mode():
    # set cuda device
    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group(backend='nccl' if dist.is_nccl_available() else 'gloo')


def clean_distributed():
    dist.destroy_process_group()


def smartprint(*msg):
    if LOCAL_RANK == 0 or LOCAL_RANK == -1:
        print(*msg)
        
        
@torch.no_grad()
def validation(config, valid_dataloader, model, use_ddp, device):
    
    result     = {}
    run_times  = 5
    n_clusters = config.dataset.class_num
    
    model.eval()
    consistency, all_specific, labels = extract_features(valid_dataloader, model, device, use_ddp)
    
    latents = {
        'consistency': consistency,
        'specificity': all_specific,
        'labels': labels
    }
    
    consistent_acc, consistent_nmi, _, concat_acc, concat_nmi, _  = test_clustering(latents, labels, n_clusters, run_times)
    result['valid-cluster-consistent-acc'] = np.mean(consistent_acc)
    result['valid-cluster-consistent-nmi'] = np.mean(consistent_nmi)
    result['valid-cluster-all-concat-acc'] = np.mean(concat_acc)
    result['valid-cluster-all-concat-nmi'] = np.mean(concat_nmi)
    
    
    consistent_acc, _, consistent_f1score, concat_acc, _, concat_f1score = test_classification(latents, labels, n_clusters, run_times)
    result['valid-classify-consistent-acc']     = np.mean(consistent_acc)
    result['valid-classify-consistent-f1score'] = np.mean(consistent_f1score)
    result['valid-classify-all-concat-acc']     = np.mean(concat_acc)
    result['valid-classify-all-concat-f1score'] = np.mean(concat_f1score)
    
    return result
    

def train_epoch(args, train_dataloader, model, epoch, device, optimizer, lr):
    """
    train the model one epoch

    Parameter
    ---------
        args : 
            _description_
        train_dataloader : 
            the train dataset dataloader
        model (_type_): _description_
        epoch (_type_): _description_
        device (_type_): _description_
        optimizer (_type_): _description_
        lr (_type_): _description_

    Returns
    -------
        _type_: _description_
    """
    
    
    losses = defaultdict(list)
    show_loss = 0.
    if args.train.use_ddp:
        model.module.train()
        parameters = list(model.module.consis_enc.parameters())
    else:
        model.train()
        parameters = list(model.consis_enc.parameters())
    if args.verbose and (LOCAL_RANK == 0 or LOCAL_RANK == -1):
        pbar = tqdm(train_dataloader, ncols=0, unit=" batch")
        
    flag = 1
    for Xs, _ in train_dataloader:
        Xs = [x.to(device) for x in Xs]
        rank = int(os.getenv('LOCAL_RANK', -1))
        
        
        if args.train.use_ddp:
            loss, loss_part = model.module.get_loss(Xs, epoch)
        else:
            loss, loss_part = model.get_loss(Xs, epoch)
            
        # backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 1)
        optimizer.step()
        optimizer.zero_grad()
        
        show_loss += loss.item()
            
        for k, v in loss_part.items():
            losses[k].append(v)  
        
        show_losses = {k: np.mean(v) for k, v in losses.items()}
        
        if args.verbose and (LOCAL_RANK == 0 or LOCAL_RANK == -1):
            loss_str = ', '.join([f'{k}:{v:.4f}' for k, v in show_losses.items()])
            pbar.set_description(f"Training | epoch: {epoch}, lr: {lr:.4f}, {loss_str}")
            pbar.update()
    
    if args.verbose and (LOCAL_RANK == 0 or LOCAL_RANK == -1):
        pbar.close()
        
    return show_losses, show_loss / len(train_dataloader)


def main(config=None):
    
    use_wandb = config.wandb
    use_ddp = config.train.use_ddp
    seed = config.seed
    runtimes = config.runtimes
    gen_intervals = config.train.gen_intervals
    valid_intervals = config.train.valid_intervals

    
    if use_ddp:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(i) for i in config.train.devices])
    
    device = get_device(config, LOCAL_RANK)
    print(f'Use: {device}')

    if use_ddp:
        init_distributed_mode()
    # for record each running.
    running_loggers = {}
    for r in range(runtimes):
        if use_ddp:
            result_dir = os.path.join(config.train.log_dir, f'c{config.consistency.c_dim}-s{config.specificity.s_dim}-seed{seed}-ddp')
        else:   
            result_dir = os.path.join(config.train.log_dir, f'c{config.consistency.c_dim}-s{config.specificity.s_dim}-seed{seed}')
        os.makedirs(result_dir, exist_ok=True)
    
    
        if use_wandb and (LOCAL_RANK == 0 or LOCAL_RANK == -1):
            wandb.init(project=config.project_name, 
                        config=config,
                        name=f"{config.experiment_name}")
        
        sub_logger = defaultdict(list)
        
        # For reproducibility
        reproducibility_setting(seed)
        
        data_transformation = get_transformation(config)
        train_dataset = get_train_dataset(config, data_transformation)
        
        if use_ddp:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                                                            train_dataset, 
                                                            shuffle=True,
                                                            drop_last=True)
            train_dataloader = DataLoader(train_dataset,
                                        num_workers=config.train.num_workers,
                                        batch_size=config.train.batch_size // WORLD_SIZE,
                                        sampler=train_sampler,
                                        pin_memory=True)
        else:
            train_dataloader = DataLoader(train_dataset,
                                        num_workers=config.train.num_workers,
                                        batch_size=config.train.batch_size,
                                        sampler=None,
                                        shuffle=True,
                                        pin_memory=True,
                                        drop_last=True)
        
        smartprint('Dataset contains {} samples views'.format(len(train_dataset), len(train_dataset[0])))
        
        # Only evaluation at the first device.
        if LOCAL_RANK == 0 or LOCAL_RANK == -1:
            eval_dataset = get_val_dataset(config, data_transformation)
            eval_dataloader = DataLoader(eval_dataset,
                                        batch_size=config.train.batch_size,
                                        num_workers=config.train.num_workers,
                                        shuffle=False,
                                        drop_last=False,
                                        pin_memory=True)
            
            
            dl = DataLoader(train_dataset, 16, shuffle=True)
            recon_samples = next(iter(dl))[0]
            recon_samples = [x.to(device, non_blocking=True) for x in recon_samples]

        if use_ddp:
            config.pseudoinputs.number = int(config.pseudoinputs.number / WORLD_SIZE)
        
        model = CL2P(config,  device=device)

        # if LOCAL_RANK == 0 or LOCAL_RANK == -1:
        #     summary(model)
        smartprint(f"Model loaded!")
        
        
        optimizer = get_optimizer(model.parameters(), config.train.lr, config.train.optim)


        # # Checkpoint
        # if config.train.resume and os.path.exists(checkpoint_path):
        #     checkpoint = torch.load(checkpoint_path, map_location='cpu')
        #     model.load_state_dict(checkpoint['model'])
        #     model = model.to(device)
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     start_epoch = checkpoint['epoch']
        #     smartprint(f"Load checkpoint {checkpoint_path} at epoch: {start_epoch}!")
        # else:
        #     start_epoch = 0
        #     model = model.to(device)
            
        start_epoch = 0
        model = model.to(device)
                    
        if use_ddp:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[LOCAL_RANK],
                output_device=LOCAL_RANK,
                find_unused_parameters=True,
                broadcast_buffers=False
            )
        
        if use_wandb and (LOCAL_RANK == 0 or LOCAL_RANK == -1):
            wandb.watch(model, log='all', log_graph=True, log_freq=10)
        
        
        if config.pseudoinputs.number != 0 and config.pseudoinputs.training_data_init: # init the pseudo-input for prior learning
            if use_ddp:
                rank    = dist.get_rank()
                torch.manual_seed(config.seed + rank)
                indices = torch.randperm(len(train_dataset))[:config.pseudoinputs.number]
                torch.manual_seed(config.seed)
            else:
                indices = torch.randperm(len(train_dataset))[:config.pseudoinputs.number]
                if len(indices) < config.pseudoinputs.number:
                    cup     = config.pseudoinputs.number - len(indices)
                    indices = torch.cat((indices, indices[:cup]))
            
            samples     = [train_dataset[i][0] for i in indices]            
            all_samples = [[samples[j][i] for j in range(config.pseudoinputs.number)] for i in range(config.dataset.views)]
            
            if use_ddp:
                model.module.init_pseudoinputs(all_samples, config.pseudoinputs.fusion, device)
            else:
                model.init_pseudoinputs(all_samples, config.pseudoinputs.fusion, device)
        
        best_loss = np.inf
        old_best_model_path = ""
        for epoch in range(start_epoch, config.train.epochs):

            lr = config.train.lr
            # Train
            if use_ddp:
                train_dataloader.sampler.set_epoch(epoch)
            losses, cur_loss = train_epoch(config, train_dataloader, model, epoch, device, optimizer, lr)
            
            # NaN
            if math.isnan(cur_loss):
                break
            
            
            # print val loss
            
            # best loss model
            if cur_loss < best_loss:
                best_loss = cur_loss
                # best_model_path = os.path.join(result_dir, f'best-{int(best_loss)}-{epoch}-{seed}.pth')
                # if old_best_model_path:
                #     # save storage.
                #     os.remove(old_best_model_path)
                # old_best_model_path = best_model_path
                
                # if config.train.use_ddp:
                #     model.module.eval()
                #     torch.save(model.module.state_dict(), best_model_path) 
                # else:
                #     model.eval()
                #     torch.save(model.state_dict(), best_model_path)
            
            
            # loss
            if use_wandb and (LOCAL_RANK == 0 or LOCAL_RANK == -1):
                wandb.log(losses, step=epoch)
            if not config.verbose:
                smartprint(f"[Training {epoch}/{config.train.epochs}]", ', '.join([f'{k}:{v:.4f}' for k, v in losses.items()]))
            
            for k, v in losses.items():
                sub_logger[k].append(v)    

            # downstream tasks
            if LOCAL_RANK == 0 or LOCAL_RANK == -1:
                if epoch % gen_intervals == 0 or epoch == (config.train.epochs-1):
                    rcons_grid = reconstruction(model, recon_samples, config.train.use_ddp)
                    sample_grid = sampling(model, config.train.samples_num, device, use_ddp)    
                    pseudoinputs_grid = pseudo_sampling(model, 24, device, use_ddp)
                    
                    if use_wandb:
                        wandb.log({'rcons-grid': wandb.Image(rcons_grid)}, step=epoch)
                        wandb.log({'conditional-samples': wandb.Image(sample_grid)}, step=epoch)    
                        wandb.log({'pseudoinputs-samples': wandb.Image(pseudoinputs_grid)}, step=epoch)

                if epoch % valid_intervals == 0 or epoch == (config.train.epochs-1):
                    if config.train.use_ddp:
                        model.module.eval()
                    else:
                        model.eval()
                        
                    print(f"[Downstream validation {epoch}/{config.train.epochs}]")
                    if config.train.use_ddp:
                        valid_result = validation(config, eval_dataloader, model, use_ddp, device)
                    else:
                        valid_result = validation(config, eval_dataloader, model, use_ddp, device)
                    if use_wandb:
                        wandb.log(valid_result, step=epoch)
                
                # Checkpoint
                # save_checkpoint(config, checkpoint_path, model, optimizer, scheduler, epoch)
            
            if use_ddp:    
                dist.barrier()
        
                
        if LOCAL_RANK == 0 or LOCAL_RANK == -1:
            if config.train.use_ddp:
                model.module.eval()
                torch.save(model.module.state_dict(), os.path.join(result_dir, "final_model.pth")) 
            else:
                model.eval()
                torch.save(model.state_dict(), os.path.join(result_dir, "final_model.pth")) 
        
            print(f"Final model saved in {result_dir}/final_model.pth")
        # update seed.        
        running_loggers[f'r{r+1}-{seed}'] = sub_logger
        seed = torch.randint(1, 9999, (1, )).item()    
        
        
        if use_wandb and (LOCAL_RANK == 0 or LOCAL_RANK == -1):
            wandb.finish()
        
        
    if LOCAL_RANK == 0 or LOCAL_RANK == -1:            
        torch.save(running_loggers, os.path.join(config.train.log_dir, 'loggers.pkl'))
        
    if use_ddp:
        clean_distributed()
    
if __name__ == '__main__':
    # Load arguments.
    
    args = parse_args()
    config = parse_config(args.config_file)
    
    main(config)
    
import warnings
warnings.filterwarnings('ignore')
import os
from collections import defaultdict
import torch
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import numpy as np
import wandb
from tqdm import tqdm
from torchinfo import summary
from scipy.optimize import linear_sum_assignment

from configs import parse_config
from models import CL2P
from sklearn import metrics
from data import get_transformation, get_train_dataset, get_val_dataset
from utils import parse_args, reproducibility_setting
from utils.visualization import plot_scatter
from utils.metrics import clustering_metric, classification_metric
from scipy.stats import zscore



@torch.no_grad()
def reconstruction(model, original):
    vspecific_recons = model(original)
    consist_recons, _ = model.consis_enc(original)
    grid = []
    for x, r in zip(original, vspecific_recons):
        grid.append(torch.cat([x, r]).detach().cpu())
    vspec_grid = make_grid(torch.cat(grid).detach().cpu())
    
    grid = []
    for x, r in zip(original, consist_recons):
        grid.append(torch.cat([x, r]).detach().cpu())
    consist_grid = make_grid(torch.cat(grid).detach().cpu())
    return consist_grid, vspec_grid


@torch.no_grad()
def extract_features(train_dataloader, model, device, use_ddp=False):
    targets = []
    consist_reprs = []
    all_vs = []
    for Xs, target in train_dataloader:
        Xs = [x.to(device) for x in Xs]
        if use_ddp:
            consist_repr_, all_v = model.module.all_features(Xs)
        else:
            consist_repr_, all_v = model.all_features(Xs)
        targets.append(target)
        consist_reprs.append(consist_repr_.detach().cpu())
        all_vs.append(all_v)
    targets = torch.concat(targets, dim=-1).numpy()
    consist_reprs = torch.vstack(consist_reprs).detach().cpu()
    all_vs = torch.vstack(all_vs).detach().cpu()
    return consist_reprs, all_vs, targets


def test_clustering(latents, labels, n_clusters, run_times):
    
    print("-"*20+"Test Clustering"+"-"*20)
    consistent_acc = []
    consistent_nmi = []
    consistent_ari = []
    concat_acc = []
    concat_nmi = []
    concat_ari = []

    for run in range(run_times):
        km = KMeans(n_clusters=n_clusters, n_init="auto")
        preds = km.fit_predict(latents['consistency'])
        acc, nmi, ari = clustering_metric(labels, preds)
        consistent_acc.append(acc)
        consistent_nmi.append(nmi)
        consistent_ari.append(ari)
        
        concat = torch.cat([latents['consistency'], latents['specificity']], dim=-1)
        km = KMeans(n_clusters=n_clusters, n_init="auto")
        preds = km.fit_predict(concat)
        acc, nmi, ari = clustering_metric(labels, preds)
        concat_acc.append(acc)
        concat_nmi.append(nmi)
        concat_ari.append(ari)
    
    print(f'[\033[33mConsistency\033[0m] acc: {100*np.mean(consistent_acc):.2f} ({100*np.std(consistent_acc):.2f}) | nmi: {100*np.mean(consistent_nmi):.2f} ({100*np.std(consistent_nmi):.2f}) | ari: {100*np.mean(consistent_ari):.2f} ({100*np.std(consistent_ari):.2f})')
    print(f'[\033[95mAll  Concat\033[0m] acc: {100*np.mean(concat_acc):.2f} ({100*np.std(concat_acc):.2f}) | nmi: {100*np.mean(concat_nmi):.2f} ({100*np.std(concat_nmi):.2f}) | ari: {100*np.mean(concat_ari):.2f} ({100*np.std(concat_ari):.2f})')

    return consistent_acc, consistent_nmi, consistent_ari, concat_acc, concat_nmi, concat_ari


def test_classification(latents, labels, n_clusters, run_times):
    
    print("-"*20+"Test Classification"+"-"*20)
    consistent_acc = []
    consistent_precision = []
    consistent_f1score = []
    concat_acc = []
    concat_precision = []
    concat_f1score = []
    for run in range(run_times):
        X_train, X_test, y_train, y_test = train_test_split(latents['consistency'], labels, test_size=0.2)
        svc = SVC()
        svc.fit(X_train, y_train)
        preds = svc.predict(X_test)
        accuracy, precision, f_score = classification_metric(y_test, preds)
        consistent_acc.append(accuracy)
        consistent_precision.append(precision)
        consistent_f1score.append(f_score)
        
        concat = torch.cat([latents['consistency'], latents['specificity']], dim=-1)
        X_train, X_test, y_train, y_test = train_test_split(concat, labels, test_size=0.2)
        svc = SVC()
        svc.fit(X_train, y_train)
        preds = svc.predict(X_test)
        accuracy, precision, f_score = classification_metric(y_test, preds)
        concat_acc.append(accuracy)
        concat_precision.append(precision)
        concat_f1score.append(f_score)
        
    print(f'[\033[33mConsistency\033[0m] acc: {100*np.mean(consistent_acc):.2f} ({100*np.std(consistent_acc):.2f}) | f1score: {100*np.mean(consistent_f1score):.2f} ({100*np.std(consistent_f1score):.2f}) | precision: {100*np.mean(consistent_precision):.2f} ({100*np.std(consistent_precision):.2f})')
    print(f'[\033[95mAll  Concat\033[0m] acc: {100*np.mean(concat_acc):.2f} ({100*np.std(concat_acc):.2f}) | f1score: {100*np.mean(concat_f1score):.2f} ({100*np.std(concat_f1score):.2f}) | precision: {100*np.mean(concat_precision):.2f} ({100*np.std(concat_precision):.2f})')

    return consistent_acc, consistent_precision, consistent_f1score, concat_acc, concat_precision, concat_f1score



def main():
    # Load arguments.
    
    args = parse_args()
    config = parse_config(args.config_file)
    device = torch.device(f'cuda:{config.train.devices[0]}')
    
    
    data_transformation = get_transformation(config)
    train_dataset = get_train_dataset(config, data_transformation)

    train_dataloader = DataLoader(train_dataset,
                                num_workers=config.train.num_workers,
                                batch_size=config.train.batch_size,
                                sampler=None,
                                shuffle=False,
                                pin_memory=True,
                                drop_last=False)
    
    run_times = 10
    n_clusters = config.dataset.class_num
    TEST = True
    visualization = False
    pic_format = 'pdf'
    model_path = config.model_path
    model = CL2P(config, device=device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    model = model.to(device)
    # print(f'Use: {device}')
    
    model.eval()
    consistency, all_specific, labels = extract_features(train_dataloader, model, device)
    
    all_concat = torch.cat([consistency, all_specific], dim=-1)
    
    latents = {
        'consistency': consistency,
        'specificity': all_specific,
        'labels': labels
    }
    torch.save(latents, os.path.join(config.train.log_dir, f'{config.dataset.name}.pkl'))
    
    
    if TEST:
        test_clustering(latents, labels, n_clusters, run_times)
        test_classification(latents, labels, n_clusters, run_times)
        
    if visualization:
        dl = DataLoader(train_dataset, 32, shuffle=True)
        recon_samples = next(iter(dl))[0]
        recon_samples = [x.to(device, non_blocking=True) for x in recon_samples]
        consist_grid, vspec_grid = reconstruction(model, recon_samples)
        save_image(consist_grid, os.path.join(config.train.log_dir, f'{config.dataset.name}-consist-recons.{pic_format}'), format=pic_format)
        save_image(vspec_grid, os.path.join(config.train.log_dir, f'{config.dataset.name}-vspec-recons.{pic_format}'), format=pic_format)
        
        
        # tsne
        tsne_samples = 2000
        
        idx = torch.rand(consistency.size(0)).argsort()
        select_idx = idx[:tsne_samples]
        
        select_consist = consistency[select_idx, :]
        select_cat = best_concat[select_idx, :]
        select_labels = labels[select_idx]
        
        from sklearn.manifold import TSNE
        from matplotlib import pyplot as plt
        
        
        _, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 4))
    
        print("Run t-sne.....")
        tsne = TSNE(n_components=2, perplexity=15, learning_rate=10)
        sz = tsne.fit_transform(select_consist.numpy())
        plot_scatter(ax1, sz, select_labels)
        handles1, ll1 = ax1.get_legend_handles_labels()
        
        bc = tsne.fit_transform(select_cat.numpy())

        plot_scatter(ax2, bc, select_labels)

        
        plt.figlegend(handles1, ll1, loc='lower center', bbox_to_anchor=(0.5, 0.96),fancybox=True, shadow=False, ncol=10, markerscale=2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(config.train.log_dir, f'{config.dataset.name}-tsne.{pic_format}'), bbox_inches='tight', format=pic_format)
        plt.close()
        
    
    
if __name__ == '__main__':
    main()
    
    
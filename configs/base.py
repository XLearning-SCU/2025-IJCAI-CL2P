
import os
from yacs.config import CfgNode as CN


# Basic config, including basic information, dataset, and training setting.
_C = CN()
# Project name, for wandb's records
_C.project_name = 'IJCAI2025'
# Project description, what problem does this project tackle?
_C.project_desc = 'A new experiment.' 
# Seed
_C.seed = 3407
# Print log
_C.verbose = True
# Enable wandb
_C.wandb = True
# Runtime
_C.runtimes = 10
# Experiment name, for wandb's records
_C.experiment_name = "None"
# Final model saved path
_C.model_path = ''
# checkpoint saved path
_C.ckpt_path = ''
# Experiment Notes
_C.note = ""


# Network setting.
_C.backbone = CN()
# Backbone architecture
_C.backbone.type = "cnn"
# Normalizations ['batch', 'layer']
_C.backbone.normalization = 'batch'
# Network init.
_C.backbone.init_method = 'kaiming'


# Dataset setting
_C.dataset = CN()
# ['Edge-Mnist', 'Edge-Fashion', 'Multi-COIL-20', 'Multi-COIL-100', 'Multi-Office-31',
# 'PolyMNIST', '']
_C.dataset.name = 'Edge-MNIST'
# Root dir
_C.dataset.root = './data/raw'
# view num
_C.dataset.views = 3
# Class num
_C.dataset.class_num = 10
# Input size
_C.dataset.crop_size = 32


# Training setting.
_C.train = CN()
# Train epoch
_C.train.epochs = 200
# Warmup step
_C.train.warmup = 0
# 
_C.train.early_stop_epochs = 50
# 
_C.train.batch_size = 128
#
_C.train.optim = 'adam'
#
_C.train.devices = [0, 1]

_C.train.lr = 0.0001

_C.train.num_workers = 2

_C.train.save_log = True
# if None, it will be set as './experiments/results/[model name]/[dataset name]'
_C.train.log_dir = ""
# the interval of evaluate epoch, defaults to 5.
_C.train.gen_intervals = 5

_C.train.valid_intervals = 5

_C.train.lr_decay_rate = 0.1

_C.train.lr_decay_epochs = 30
# samling num.
_C.train.samples_num = 6
# using checkpoint training.
_C.train.resume = False

_C.train.ckpt_path = ""

_C.train.use_ddp = True
# The parameter for curriculum adapter, default 2.0
_C.train.alpha = 2.0
# the curriculum strategy, ["power", "cosine", "linear", "exponent"]
_C.train.scheduler = "power"

# for consistency encoder setting.
_C.consistency = CN()
# 
_C.consistency.enable = True
# 
_C.consistency.continuous = True
# consistency bottleneck dim.
_C.consistency.c_dim = 32
# 
_C.consistency.in_channel = 1
# 
_C.consistency.ch_mult = [1, 2, 4, 8]
# 
_C.consistency.block_size = 8
#
_C.consistency.basic_hidden_dim = 32
#
_C.consistency.latent_ch = 10
#
_C.consistency.num_res_blocks = 3
#
_C.consistency.temperature = 0.5
#
_C.consistency.kld_weight = 1.0
# for categories vae
_C.consistency.alpha = 1.0

_C.consistency.anneal = 0


# commom feature pooling method. mean, sum, or first
_C.fusion = CN()
# ['Mixture-of-Experts','Product-of-Experts','concat']
_C.fusion.type = 'moe'

_C.fusion.pooling_method = 'sum'
# view specific fusion weight
_C.fusion.vs_weights = 1.


# for view-specific encoder setting.
_C.specificity = CN()
# 
_C.specificity.enable = True
# Basic hidden dim
_C.specificity.basic_hidden_dim = 32
# Each layer output channel, will multiply with basic hidden dim.
_C.specificity.ch_mult = [1, 2, 4, 8]
# Image shape // 2 * len(ch_mult), for example, if the image shape is 64x64, and the 
# len(ch_mult) is 8, then we have the block_size = 64/2*4 = 8.
_C.specificity.block_size = 8
# encoder output channel.
_C.specificity.latent_ch = 10
# s latent dim
_C.specificity.s_dim = 32

_C.specificity.kld_weight = 1.0
# num_res_blocks for the number of the Encoder and Decoder's residual block.
_C.specificity.num_res_blocks = 2


# for learnable prior setting
_C.pseudoinputs = CN()

_C.pseudoinputs.number = 500

_C.pseudoinputs.input_size = [3, 64, 64]

_C.pseudoinputs.training_data_init = True

_C.pseudoinputs.mean = 0.5

_C.pseudoinputs.std = 0.02

_C.pseudoinputs.nonlinear = 1

_C.pseudoinputs.GTM = False

_C.pseudoinputs.grid_size = 50.0

_C.pseudoinputs.mask_ratio = 0.0

_C.pseudoinputs.mask_patch = 2

_C.pseudoinputs.mask_value = 0.0

_C.pseudoinputs.blur = False

_C.pseudoinputs.fusion = 0.


# disentanglement
_C.disent = CN()

_C.disent.enable = True

_C.disent.lam = 1.0

_C.disent.hidden_size = 100
# for consistency dim.
_C.disent.alpha = 1.0

_C.disent.mode = 'bias'


def parse_config(config_file_path):
    """
    Initialize configuration.
    """
    config = _C.clone()
    # merge specific config.
    config.merge_from_file(config_file_path)
    
    
    if hasattr(config, "experiment_name"):
        experiment_name = config.experiment_name.replace("${", "{")
        config.experiment_name = experiment_name.format(
            dataset = config.dataset
        )
    
    # Replace placeholders in paths
    if hasattr(config, "model_path"):
        model_path = config.model_path.replace("${", "{")
        config.model_path = model_path.format(
            dataset=config.dataset,
            consistency=config.consistency,
            specificity=config.specificity,
            seed=config.seed
        )
        
    if not config.train.log_dir:
        path = f'./experiments/{config.experiment_name}'
        os.makedirs(path, exist_ok=True)
        config.train.log_dir = path
    else:
        os.makedirs(config.train.log_dir, exist_ok=True)
    # config.freeze()
    return config
import torch
import argparse
import numpy as np


def save_checkpoint(config, checkpoint_path, model, optimizer, scheduler, epoch):
    checkpoint_state_dict = {
        'optimizer': optimizer.state_dict(),
        'epoch': epoch+1,
    }

    if config.train.use_ddp:
        checkpoint_state_dict['model'] = model.module.state_dict()
    else:
        checkpoint_state_dict['model'] = model.state_dict()

    if scheduler is not None:
        checkpoint_state_dict['scheduler'] = scheduler.state_dict()

    # Checkpoint
    torch.save(checkpoint_state_dict, checkpoint_path)


def load_checkpoint():
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file', '-f', type=str, help='Config File')
    args = parser.parse_args()
    return args

        

def reproducibility_setting(seed):
    """
    set the random seed to make sure reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)

    print('Global seed:', seed)


def get_device(args, local_rank):
    if args.train.use_ddp:
        device = torch.device(
            f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device(f"cuda:{args.train.devices[0]}") if torch.cuda.is_available(
        ) else torch.device('cpu')
    return device


def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]


def label_to_one_hot(label_idx, num_classes) -> torch.Tensor:
    return torch.nn.functional.one_hot(label_idx,
                                       num_classes=num_classes)


def one_hot_to_label(one_hot_arr: torch.Tensor) -> torch.Tensor:
    return one_hot_arr.argmax(dim=1)


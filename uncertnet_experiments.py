import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from types import SimpleNamespace
from tqdm import tqdm
from uncertnet import dataset, uncertnet

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def get_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--placeholder", default="yeet", help="I do nothing", type=str)
    return parser.parse_args()

def get_config(args):
    config = SimpleNamespace()
    # Debug
    config.overfit_datalim = 2_000_000
    # Paths
    config.uncertnet_data_path = "auto_UPDRS/data/body/h36m/uncertnet/"
    config.uncertnet_file_pref = "h36m_"
    config.cam_ids_file = '_cam_ids.npy'
    config.pred_poses_3d_file = '_pred_poses.npy'
    config.pred_camrots_file = '_pred_rots.npy'
    config.triang_poses_3d_file = '_triang_poses.npy'
    config.gt_poses_3d_file = '_gt_poses.npy'
    # Data format
    config.err_scale = 1000   # Scale the err by this much to make it easier to train
    config.num_kpts = 15
    # Model format
    config.use_confs = False
    config.out_per_kpt = False
    if config.out_per_kpt:
        config.out_dim = config.num_kpts
    else:
        config.out_dim = 1
    # Training
    config.val_split = 0.1
    config.test_split = 0   # Now I have explicit test file of fixed subjects. Set = 0 to split train data into train/val only
    config.epochs = 5
    config.batch_size = 4096
    config.lr = 5e-4
    # config.weight_decay = 1e-5
    # Logging
    config.b_print_freq = 100
    config.e_print_freq = 1
    config.uncertnet_ckpt_path = "auto_UPDRS/model_checkpoints/uncertnet/uncert_net_bestval.pth"
    config.uncertnet_save_ckpts = False
    # Eval
    config.num_cams = 4
    config.eval_batch_size = 2
    # Misc
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return config


def main():
    args = get_args()
    config = get_config(args)

    # Do some stuff
    model = uncertnet.uncert_net_wrapper(config)
    # model.train()
    model.evaluate()
    
if __name__ == '__main__':
    main()
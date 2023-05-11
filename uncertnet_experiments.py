import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from types import SimpleNamespace
from tqdm import tqdm
import wandb
from uncertnet import dataset, uncertnet

# Reproducibility
# SEED = 42
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)


def get_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--placeholder", default="yeet", help="I do nothing", type=str)
    return parser.parse_args()

def init_logging(args, config):
    wandb.init(
        # set the wandb project where this run will be logged
        project="Bootstrapping 3D pose estimation with multi view data",
        # track hyperparameters and run metadata
        config={
            # Architecture
            "use_confs": config.use_confs,
            "use_camID": config.use_camID,
            "out_per_kpt": config.out_per_kpt,
            "out_directional": config.out_directional,
            "hidden_dim": config.hidden_dim,
            # Training
            "lr": config.lr,
            "epochs": config.epochs,
            "gt_targets": config.use_gt_targets,
            # Eval
            "cams": config.cams,
            "num_cams": config.num_cams,
            }
        )
    return wandb

def get_config(args):
    config = SimpleNamespace()
    # Logging -----------------------
    config.log = True
    # -------------------------------
    # Tasks/Experiments
    config.num_runs = 3
    config.train = True
    config.eval = True

    # Logging Details
    config.b_print_freq = 100
    config.e_print_freq = 1
    config.uncertnet_ckpt_path = "auto_UPDRS/model_checkpoints/uncertnet/uncert_net_bestval.pth"
    config.uncertnet_save_ckpts = True

    # Model Architecture
    config.simple_linear = False

    config.hidden_dim = 8

    config.use_confs = False        # use 3D pred confidences?
    config.use_camID = False

    config.out_per_kpt = True       # output per-kpt err vector?
    config.out_directional = True   # output directional err vector?
    config.num_kpts = 15

    if config.out_per_kpt:
        config.out_dim = config.num_kpts
    else:
        config.out_dim = 1
    if config.out_directional:
            config.out_dim *= 3

    # Data format
    config.use_gt_targets = False   # use GT 2D/3D poses as targets? and GT-based backbone?
    config.cams = [
                    0, 
                    1,
                    2,
                    3
                ]  # All Cam IDs: [0, 1, 2, 3]
    config.num_cams = len(config.cams)
    config.err_scale = 25   # Scale the err by this much to make it easier to train?
    # Training
    config.val_split = 0.3
    config.test_split = 0   # Now I have explicit test file of fixed subjects. Set = 0 to split train data into train/val only
    config.epochs = 8
    config.batch_size = 4096
    config.lr = 1e-3    # 1e-3
    if config.simple_linear == True:
        config.lr = 1e-2

    config.use_step_lr = False
    config.step_lr_size = 1
    # Evaluation
    config.eval_batch_size = 4096
    # Model Paths
    config.uncertnet_data_path = "auto_UPDRS/data/body/h36m/uncertnet/"
    
    if config.use_gt_targets:
        config.lifter_ckpt_path = "auto_UPDRS/model_checkpoints/body_pose/model_lifter_gt2d_gt3d.pt"
        config.uncertnet_data_path += "gt2d_gt3d/"
    else:
        config.lifter_ckpt_path = "auto_UPDRS/model_checkpoints/body_pose/model_lifter_ap2d_ap3d.pt"
    
    # Data Paths
    config.uncertnet_file_pref = "h36m_"
    config.cam_ids_file = '_cam_ids.npy'
    config.ap_pred_poses_2d_file = '_ap_preds.npy'
    config.pred_poses_3d_file = '_pred_poses.npy'
    config.pred_camrots_file = '_pred_rots.npy'
    config.triang_poses_3d_file = '_triang_poses.npy'
    config.gt_poses_3d_file = '_gt_poses.npy'
    # Misc
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.overfit_datalim = None
    return config


def main():
    args = get_args()
    config = get_config(args)

    # Do some stuff
    for i in range(config.num_runs):
        print("Run: {} of {}".format(i+1, config.num_runs))
        logger = init_logging(args, config) if config.log else None
        model = uncertnet.uncert_net_wrapper(config, logger)
        # Do tasks as desired
        if config.train: model.train()
        if config.eval: model.evaluate()
        if config.log: logger.finish()
        
if __name__ == '__main__':
    main()
import torch
import torch.nn
import torch.optim
import numpy as np
from torch.utils import data
import torch.optim as optim
import model_confidences
from types import SimpleNamespace
from pytorch3d.transforms import so3_exponential_map as rodrigues
from numpy.random import default_rng

from data.body.body_dataset import PD_AP_Dataset, get_2D_data
from utils.pose_utils import procrustes_torch
from utils.helpers import print_losses
from utils.metrics import loss_weighted_rep_no_scale
import utils.info as info


def get_config():
    config = SimpleNamespace()
    # Tasks
    config.updrs_tasks = 'free_form_oval'
    # config.tasks = ['train']
    # config.tasks = ['eval']
    config.tasks = ['train', 'eval']

    # Training
    # config.train_subjs = info.subjects_new_sys
    config.train_subjs = ['S01', 'S28', 'S29',]
    config.val_subjs = ['S31']

    config.learning_rate = 0.0001
    config.train_bs = 256
    config.eval_bs = 4096
    config.N_epochs = 30
    config.NoEval = True
    # weights for the different losses
    config.weight_rep = 1
    config.weight_view = 1
    config.weight_camera = 0.1

    # Paths
    config.root_path = 'auto_UPDRS/'
    config.data_path = config.root_path + 'data/body/2d_proposals/'
    config.datafile = config.data_path + 'free_form_oval_2D_kpts-DEBUG.pickle'
    config.lifter_ckpt_path = config.root_path + 'model_checkpoints/body_pose/PD/PD_3D_lifter'
    
    # Data format
    config.num_kpts = 15
    config.all_cams = ['cam0', 'cam1']
    return config

def view_cons_and_rep_loss(config, sample, inp_poses_2d_rs, inp_confs_2d, 
                           pred_rot_rs, pred_poses_rs,
                           rot_poses_rs, confidences_rs,):
    '''
    This is a mess, but it's the same as Mohsens code.
    '''
    # view & camera consistency computed in same loop
    loss_view_cons = 0
    loss_rep = 0
    for c_cnt in range(len(config.all_cams)):
        # get all cameras and active cameras
        ac = np.array(range(len(config.all_cams)))
        coi = np.delete(ac, c_cnt)
        # view consistency
        projected_to_other_cameras = pred_rot_rs[:, coi].matmul(pred_poses_rs.reshape(-1, len(config.all_cams), 3, config.num_kpts)[:, c_cnt:c_cnt+1].repeat(1, len(config.all_cams)-1, 1, 1)).reshape(-1, len(config.all_cams)-1, config.num_kpts*3)
        loss_view_cons += loss_weighted_rep_no_scale(inp_poses_2d_rs.reshape(-1, len(config.all_cams), config.num_kpts*2)[:, coi].reshape(-1, config.num_kpts*2),
                                                projected_to_other_cameras.reshape(-1, config.num_kpts*3),
                                                inp_confs_2d.reshape(-1, len(config.all_cams), config.num_kpts)[:, coi].reshape(-1, config.num_kpts))
        # camera consistency
        relative_rotations = pred_rot_rs[:, coi].matmul(pred_rot_rs[:, [c_cnt]].permute(0, 1, 3, 2))
        # only shuffle in between subjects
        rng = default_rng()
        for subject in sample['subjects'].unique():
            # only shuffle if enough subjects are available
            if (sample['subjects'] == subject).sum() > 1:
                shuffle_subjects = (sample['subjects'] == subject)
                num_shuffle_subjects = shuffle_subjects.sum()
                rand_perm = rng.choice(num_shuffle_subjects.cpu().numpy(), size=num_shuffle_subjects.cpu().numpy(), replace=False)
                samp_relative_rotations = relative_rotations[shuffle_subjects]
                samp_rot_poses_rs = rot_poses_rs[shuffle_subjects]
                samp_inp_poses = inp_poses_2d_rs[shuffle_subjects][:, coi].reshape(-1, config.num_kpts*2)
                samp_inp_confidences = confidences_rs[shuffle_subjects][:, coi].reshape(-1, config.num_kpts)

                random_shuffled_relative_projections = samp_relative_rotations[rand_perm].matmul(samp_rot_poses_rs.reshape(-1, len(config.all_cams), 3, config.num_kpts)[:, c_cnt:c_cnt+1].repeat(1, len(config.all_cams)-1, 1, 1)).reshape(-1, len(config.all_cams)-1, config.num_kpts*3)

                loss_rep += loss_weighted_rep_no_scale(samp_inp_poses,
                                                            random_shuffled_relative_projections.reshape(-1, config.num_kpts*3),
                                                            samp_inp_confidences)


    return loss_view_cons, loss_rep

def train(config):
    print("--- Training ---")
    # setup datasets
    poses_2d_train, confs_2d_train, out_subject_train = get_2D_data(config.train_subjs, config.updrs_tasks, 
                                                            config.datafile, normalized=True, mohsens_output=True)
    train_set = PD_AP_Dataset(poses_2d_train, confs_2d_train, out_subject_train)
    train_loader = data.DataLoader(train_set, batch_size=config.train_bs, shuffle=True, num_workers=0)

    # Define lifting network & optim
    model = model_confidences.Lifter().cuda()

    optimizer = optim.Adam(list(model.parameters()), lr=config.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

    losses = SimpleNamespace()
    losses_mean = SimpleNamespace()

    for epoch in range(config.N_epochs):
        for i, sample in enumerate(train_loader):
            # not the most elegant way to extract the dictionary
            poses_2d = {key:sample[key] for key in config.all_cams}
            inp_poses_2d = torch.zeros((poses_2d['cam0'].shape[0] * len(config.all_cams), config.num_kpts*2)).cuda()
            inp_confs_2d = torch.zeros((poses_2d['cam0'].shape[0] * len(config.all_cams), config.num_kpts)).cuda()

            # poses_2d is a dictionary. It needs to be reshaped to be propagated through the model.
            cnt = 0
            for b in range(poses_2d['cam0'].shape[0]):
                for c_idx, cam in enumerate(poses_2d):
                    inp_poses_2d[cnt] = poses_2d[cam][b]
                    inp_confs_2d[cnt] = sample['confidences'][c_idx][b]
                    cnt += 1
    
            # forward pass
            pred = model(inp_poses_2d, inp_confs_2d)
            pred_poses = pred[0]
            pred_cam_angles = pred[1]
            
            # use Rodrigues formula (Equations 3 and 4) to convert predicted axis angle notation angles to rotation matrix
            pred_rot = rodrigues(pred_cam_angles)
            
            # reproject to original cameras after applying rotation to the canonical poses
            rot_poses = pred_rot.matmul(pred_poses.reshape(-1, 3, config.num_kpts))
            rot_poses = rot_poses.reshape(-1, config.num_kpts*3)

            # reprojection loss
            losses.rep = loss_weighted_rep_no_scale(inp_poses_2d, rot_poses, inp_confs_2d)

            # reshape things and compute view & cam consistency losses
            pred_poses_rs = pred_poses.reshape((-1, len(config.all_cams), config.num_kpts*3))
            pred_rot_rs = pred_rot.reshape(-1, len(config.all_cams), 3, 3)
            confidences_rs = inp_confs_2d.reshape(-1, len(config.all_cams), config.num_kpts)
            inp_poses_2d_rs = inp_poses_2d.reshape(-1, len(config.all_cams), config.num_kpts*2)
            rot_poses_rs = rot_poses.reshape(-1, len(config.all_cams), config.num_kpts*3)

            losses.view, losses.camera = view_cons_and_rep_loss(config, sample, inp_poses_2d_rs, inp_confs_2d, 
                                                                pred_rot_rs, pred_poses_rs, 
                                                                rot_poses_rs, confidences_rs,)
            # Loss and backprop
            losses.loss = config.weight_rep * losses.rep + \
                        config.weight_view * losses.view + \
                        config.weight_camera * losses.camera
            
            optimizer.zero_grad()
            losses.loss.backward()
            optimizer.step()

            for key, value in losses.__dict__.items():
                if key not in losses_mean.__dict__.keys():
                    losses_mean.__dict__[key] = []
                losses_mean.__dict__[key].append(value.item())

            # print progress every 100 iterations
            if not i % 100:
                # print the losses to the console
                print_losses(epoch, i, len(train_set) / config.train_bs, losses_mean.__dict__, print_keys=not(i % 1000))
                # this line is important for logging!
                losses_mean = SimpleNamespace()

        # save the new trained model every epoch
        torch.save(model.state_dict(), config.lifter_ckpt_path)
        print("  Model Saved")

        scheduler.step()
    print('done training.\n')

def eval(config):
    print("--- Eval ---")
    # load the model
    model = model_confidences.Lifter().cuda()
    model.load_state_dict(torch.load(config.lifter_ckpt_path))
    model.eval()

    poses_2d_val, confs_2d_val, out_subject_val = get_2D_data(config.val_subjs, config.updrs_tasks, 
                                                            config.datafile, normalized=True, mohsens_output=True)
    val_set = PD_AP_Dataset(poses_2d_val, confs_2d_val, out_subject_val)
    val_loader = data.DataLoader(val_set, batch_size=config.eval_bs, shuffle=False, num_workers=0)

    losses = SimpleNamespace()
    losses_mean = SimpleNamespace()
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            # not the most elegant way to extract the dictionary
            poses_2d = {key:sample[key] for key in config.all_cams}
            inp_poses_2d = torch.zeros((poses_2d['cam0'].shape[0] * len(config.all_cams), config.num_kpts*2)).cuda()
            inp_confs_2d = torch.zeros((poses_2d['cam0'].shape[0] * len(config.all_cams), config.num_kpts)).cuda()

            # poses_2d is a dictionary. It needs to be reshaped to be propagated through the model.
            cnt = 0
            for b in range(poses_2d['cam0'].shape[0]):
                for c_idx, cam in enumerate(poses_2d):
                    inp_poses_2d[cnt] = poses_2d[cam][b]
                    inp_confs_2d[cnt] = sample['confidences'][c_idx][b]
                    cnt += 1
    
            # forward pass
            pred = model(inp_poses_2d, inp_confs_2d)
            pred_poses = pred[0]
            pred_cam_angles = pred[1]
            
            # use Rodrigues formula (Equations 3 and 4) to convert predicted axis angle notation angles to rotation matrix
            pred_rot = rodrigues(pred_cam_angles)
            
            # reproject to original cameras after applying rotation to the canonical poses
            rot_poses = pred_rot.matmul(pred_poses.reshape(-1, 3, config.num_kpts))
            rot_poses = rot_poses.reshape(-1, config.num_kpts*3)

            # reprojection loss
            losses.rep = loss_weighted_rep_no_scale(inp_poses_2d, rot_poses, inp_confs_2d)

            # reshape things and compute view & cam consistency losses
            pred_poses_rs = pred_poses.reshape((-1, len(config.all_cams), config.num_kpts*3))
            pred_rot_rs = pred_rot.reshape(-1, len(config.all_cams), 3, 3)
            confidences_rs = inp_confs_2d.reshape(-1, len(config.all_cams), config.num_kpts)
            inp_poses_2d_rs = inp_poses_2d.reshape(-1, len(config.all_cams), config.num_kpts*2)
            rot_poses_rs = rot_poses.reshape(-1, len(config.all_cams), config.num_kpts*3)

            losses.view, losses.camera = view_cons_and_rep_loss(config, sample, inp_poses_2d_rs, inp_confs_2d, 
                                                                pred_rot_rs, pred_poses_rs, 
                                                                rot_poses_rs, confidences_rs,)
            # Loss and backprop
            losses.loss = config.weight_rep * losses.rep + \
                        config.weight_view * losses.view + \
                        config.weight_camera * losses.camera

            for key, value in losses.__dict__.items():
                if key not in losses_mean.__dict__.keys():
                    losses_mean.__dict__[key] = []
                losses_mean.__dict__[key].append(value.item())

            # print progress every 100 iterations
            if not i % 100:
                # print the losses to the console
                print_losses(0, i, len(val_set) / config.train_bs, losses_mean.__dict__, print_keys=not(i % 1000))
                # this line is important for logging!
                losses_mean = SimpleNamespace()

def main():
    config = get_config()

    if 'train' in config.tasks: train(config)
    if 'eval' in config.tasks: eval(config)

if __name__ == '__main__':
    main()  

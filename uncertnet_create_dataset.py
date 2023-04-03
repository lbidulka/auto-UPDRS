import torch
import torch.nn
import torch.optim
import numpy as np
from torch.utils import data
# from utils.data import *
from utils.luke_utils.dataset import H36MDataset
import model_confidences
from types import SimpleNamespace
import pytorch3d.transforms as transform
from utils.camera import *
from utils.loss import *
from utils.plot import *
from utils.correct_action import *
from utils.epipolar import *
import copy
import argparse
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument("--lifter_ckpt_path", default="models/model_lifter.pt", help="ckpt for 3d lifter model", type=str)
    # Data
    parser.add_argument("--h36m_path", default="data/", help="h36m dataset location", type=str)
    return parser.parse_args()

def load_data_files(config):
    print("Loading data files...", end='')
    triang_3d = np.load(config.triang_3d_data_path, allow_pickle=True)
    triang_3d = triang_3d['positions_3d'].item()
    gt_3d = np.load(config.gt_3d_data_path,allow_pickle=True)
    gt_3d = gt_3d['positions_3d'].item()
    ap_2d = np.load(config.ap_2d_data_path, allow_pickle=True)
    ap_2d = ap_2d['positions_2d'].item()
    print("done.")
    return triang_3d, gt_3d, ap_2d

def get_gt_bone_len(gt_3d):
    bone_length=[]
    for subject in gt_3d.keys():
        if subject!='S9' and subject!='S11':
            for action in gt_3d[subject].keys():
                anim = gt_3d[subject][action][0]
                bone_length.append(np.mean(np.linalg.norm(anim[:,0,:] - anim[:,7,:], axis=-1)))
    bone_real = np.mean(bone_length) 
    return bone_real

def combine_gt_triang_and_get_triang_err(config, gt_3d, triang_3d, bone_real):
    num_joints = config.num_kpts
    triang_err = 0
    N = 0
    # TODO: WAS THIS INTENTIONALLY EDITING THE GT?
    for subject in gt_3d.keys():
        for action in gt_3d[subject].keys():
            anim_triang = triang_3d[subject][action]['positions_triang']
            anim = gt_3d[subject][action]
            positions_3d = []
            positions_3d_triang=[]

            for ii, cam in enumerate(range(4)):
                pos_3d = anim[ii]
                pos_3d -= pos_3d[:, 0:1,:] # Remove global offset, but keep trajectory in first position
                
                pos_3d_triang = anim_triang[ii] - anim_triang[ii][:, 0:1,:]
                bone_triang = np.mean(np.linalg.norm(pos_3d_triang[0,0,:] - pos_3d_triang[0,7,:], axis=-1))
                pos_3d_triang = pos_3d_triang * bone_real / bone_triang                
                pos_3d_triang -= pos_3d_triang[:, 0:1, :]
                                            
                positions_3d.append(pos_3d)
                positions_3d_triang.append(pos_3d_triang)

                # TODO: WHY MOHSEN ONLY USED LIMITED SUBJECTS?
                # if subject in ['S1','S5','S6','S7','S8',]: 
                if subject in config.subjects:
                    n = len(pos_3d_triang) * num_joints
                    if subject == 'S5' and action in ['Sitting', 'Sitting 1']:
                        pass
                    else:
                        N += n
                        triang_err += numpy_nmpjpe(pos_3d_triang, pos_3d) * 1000 * n

            # TODO: THIS LOOKS LIKE WHERE THE DATA IS GETTING EDITED AND COMBINED
            # TODO: WHY IS SAME DATA GETTING PUT FOR BOTH POSITIONS_3D AND POSITIONS_TRIANG????
            anim['positions_gt_3d'] = positions_3d
            # anim['positions_triang'] = positions_3d
            anim['positions_triang_3d'] = positions_3d_triang

    print('Triangulation Error', triang_err/N)
    return gt_3d

def clean_2d_detections(ap_2d, kpts_3d):
    corr_ap_2d = copy.deepcopy(ap_2d)
    for subject in kpts_3d.keys():
        assert subject in ap_2d, 'Subject {} is missing from the 2D detections dataset'.format(subject)
        for action in kpts_3d[subject].keys():
            action_corrected = correct_action[subject][action]
            assert action_corrected in ap_2d[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(action, subject)
            if 'positions' not in kpts_3d[subject][action]:
                continue
                
            for cam_idx in range(len(ap_2d[subject][action_corrected])):
                # We check for >= instead of == because some videos in H3.6M contain extra frames
                mocap_length = kpts_3d[subject][action]['positions_gt_3d'].shape[0]
                
                if ap_2d[subject][action_corrected][cam_idx].shape[0] > mocap_length:
                    # Shorten sequence
                    corr_ap_2d[subject][action_corrected][cam_idx] = ap_2d[subject][action_corrected][cam_idx][:mocap_length]
    
    return corr_ap_2d

def norm_2d_kpts(config, kpts_3d, corr_ap_2d):
    num_joints = config.num_kpts
    confidences = {}
    for subject in kpts_3d.keys():
        confidences[subject] = {}
        assert subject in corr_ap_2d, 'Subject {} is missing from the 2D detections dataset'.format(subject)
        for action in kpts_3d[subject].keys():
            action_corrected = correct_action[subject][action]
            confidences[subject][action_corrected] = []
            kps_confs = corr_ap_2d[subject][action_corrected]

            for cam_idx in range(4):
                # Normalize camera frame
                confidences[subject][action_corrected].append([])
                kps = copy.deepcopy(kps_confs)
                conf = copy.deepcopy(kps_confs) # TODO: ???

                kps = kps[cam_idx][:,:,:2]
                conf = np.ones((kps.shape[0],15))

                kps = kps - kps[:,0:1]
                kps = np.transpose(kps, [0,2,1])
                kps = kps.reshape(-1, num_joints*2)
                kps = kps / (np.linalg.norm(kps, ord=2, axis=1, keepdims=True) + 0.0001)

                corr_ap_2d[subject][action_corrected][cam_idx] = kps
                confidences[subject][action_corrected][cam_idx] = conf
    
    return corr_ap_2d, confidences
    
# def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
    #     out_poses_3d = []
    #     out_poses_2d = []
    #     out_conf = []
    #     for subject in subjects:
    #         for action in gt_3d[subject].keys():
    #             action_corrected=correct_action[subject][action]
    #             if action_filter is not None:
    #                 found = False
    #                 for a in action_filter:
    #                     if action.startswith(a):
    #                         found = True
    #                         break
    #                 if not found:
    #                     continue
                    
    #             poses_2d = ap_2d[subject][action_corrected]
    #             conf = confidences[subject][action_corrected]
                
    #             for i in range(len(poses_2d)): # Iterate across cameras
    #                 out_poses_2d.append(poses_2d[i])
    #                 out_conf.append(conf[i])
                    
    #             if parse_3d_poses and 'positions_3d' in gt_3d[subject][action]:
    #                 poses_3d = gt_3d[subject][action]['positions_3d']
    #                 assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
    #                 for i in range(len(poses_3d)): # Iterate across cameras
    #                     out_poses_3d.append(poses_3d[i])

    #     return  np.concatenate(out_poses_2d, axis=0), np.concatenate(out_conf, axis=0), np.concatenate(out_poses_3d, axis=0)

def fetch_data(subjects, kpts_3d, normed_ap_2d_kpts, ap_2d_confs,
                action_filter=None, parse_3d_poses=True):
        out_subject = []
        out_poses_2d = []
        out_triang_poses_3d = []
        out_gt_poses_3d = []
        out_conf= []
        for i in range(4):
            out_poses_2d.append([])
            out_triang_poses_3d.append([])
            out_gt_poses_3d.append([])
            out_conf.append([])
        for subject in subjects:
            for action in kpts_3d[subject].keys():
                action_corrected = correct_action[subject][action]
                # action_corrected=action
                if action_filter is not None:
                    found = False
                    for a in action_filter:
                        if action.startswith(a):
                            found = True
                            break
                    if not found:
                        continue
                if subject=='S5':
                    if action in ['Sitting','Sitting 1']:
                        continue    
                poses_2d = normed_ap_2d_kpts[subject][action_corrected]
                conf = ap_2d_confs[subject][action_corrected]

                for i in range(len(poses_2d)): # Iterate across cameras
                    out_poses_2d[i].append(poses_2d[i])
                    out_conf[i].append(conf[i])

                if parse_3d_poses and 'positions_triang_3d' in kpts_3d[subject][action]:
                    poses_3d = kpts_3d[subject][action]['positions_triang_3d']   # TODO: WHY DID THIS SEEM TO USE GT?
                    assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                    for i in range(len(poses_3d)): # Iterate across cameras
                        out_triang_poses_3d[i].append(poses_3d[i])
                
                if parse_3d_poses and 'positions_gt_3d' in kpts_3d[subject][action]:
                    poses_3d = kpts_3d[subject][action]['positions_gt_3d']   # TODO: WHY DID THIS SEEM TO USE GT?
                    assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                    for i in range(len(poses_3d)): # Iterate across cameras
                        out_gt_poses_3d[i].append(poses_3d[i])

                out_subject.append(np.ones(len(poses_2d[0])) * int(subject[-1]))
                    
        for i in range(len(poses_2d)): # Iterate across cameras
            out_poses_2d[i] = np.concatenate(out_poses_2d[i], axis=0) 
            out_conf[i] = np.concatenate(out_conf[i], axis=0)   
            out_triang_poses_3d[i] = np.concatenate(out_triang_poses_3d[i], axis=0)
            out_gt_poses_3d[i] = np.concatenate(out_gt_poses_3d[i], axis=0)
        out_subject = np.concatenate(out_subject, axis=0)

        return out_triang_poses_3d, out_gt_poses_3d, out_poses_2d, out_conf, out_subject

def get_config(args):
    config = SimpleNamespace()
    # Paths
    config.data_path = 'data/'
    config.triang_3d_data_path = 'data/triangulated_3d_h36m_ap.npz'
    config.gt_3d_data_path = 'data/data_3d_h36m_ap.npz'
    config.ap_2d_data_path = 'data/data_2d_h36m_ap.npz'
    # Data format
    config.num_kpts = 15
    config.subjects = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
    config.train_subjects = ['S1', 'S5', 'S6', 'S7', 'S8']
    config.test_subjects = ['S9', 'S11']
    config.all_cams = ['cam0', 'cam1', 'cam2', 'cam3']
    config.all_cams_3d = ['cam0_3d_gt', 'cam1_3d_gt', 'cam2_3d_gt', 'cam3_3d_gt', 
                          'cam0_3d_tr', 'cam1_3d_tr', 'cam2_3d_tr', 'cam3_3d_tr']
    # Misc
    config.batch_size = 1024
    config.save_outputs = True
    return config

def main():
    input_args = get_args()
    config = get_config(input_args)

    num_joints = config.num_kpts

    triang_3d, gt_3d, ap_2d = load_data_files(config)

    # Do some cleaning + normalization on the data
    bone_real = get_gt_bone_len(gt_3d)
    print('bone_real:', bone_real)     
    kpts_3d = combine_gt_triang_and_get_triang_err(config, gt_3d, triang_3d, bone_real)
    corr_ap_2d = clean_2d_detections(ap_2d, kpts_3d)
    normed_ap_2d_kpts, ap_2d_confs = norm_2d_kpts(config, kpts_3d, corr_ap_2d)

    # Make it into a dataset
    data_3d_triang, data_3d_gt, data_2d_ap, data_2d_ap_conf, data_subj = fetch_data(config.train_subjects, kpts_3d, normed_ap_2d_kpts, ap_2d_confs, 
                                                                            action_filter=None, parse_3d_poses=True)
    train_dataset = H36MDataset(data_3d_triang, data_3d_gt, data_2d_ap, data_2d_ap_conf, data_subj)
    train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    test_3d_triang, test_3d_gt, test_2d_ap, test_2d_ap_conf, test_subj = fetch_data(config.test_subjects, kpts_3d, normed_ap_2d_kpts, ap_2d_confs, 
                                                                            action_filter=None, parse_3d_poses=True)
    test_dataset = H36MDataset(test_3d_triang, test_3d_gt, test_2d_ap, test_2d_ap_conf, test_subj)
    test_loader = data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    # Setup pretrained lifting network
    pose_lifter = model_confidences.Lifter().cuda()
    checkpoint = torch.load('models/model_lifter.pt')
    pose_lifter.load_state_dict(checkpoint.state_dict())
    pose_lifter.eval()


    print(" ---- TRAIN SET ----")
    out_cam_ids = []
    out_pred_poses = []
    out_pred_rots = []
    out_triang_poses = []
    out_gt_poses = []
    vanilla_errs = []
    triangulated_errs = []
    for i, sample in enumerate(tqdm(train_loader)):
        # print(sample.keys())
        _poses_2d = {key:sample[key] for key in config.all_cams}
        _poses_3d = {key:sample[key] for key in config.all_cams_3d}

        inp_poses_2d = torch.zeros((_poses_2d['cam0'].shape[0] * len(config.all_cams), num_joints*2)).cuda()
        poses_3d_tr = torch.zeros((_poses_3d['cam0_3d_tr'].shape[0] * len(config.all_cams), num_joints,3)).cuda()
        poses_3d_gt = torch.zeros((_poses_3d['cam0_3d_gt'].shape[0] * len(config.all_cams), num_joints,3)).cuda()
        inp_confs_2d = torch.zeros((_poses_2d['cam0'].shape[0] * len(config.all_cams), num_joints)).cuda()

        cam_ids = torch.zeros((_poses_2d['cam0'].shape[0] * len(config.all_cams))).cuda()

        cnt = 0
        for b in range(_poses_2d['cam0'].shape[0]):
            for c_idx, cam in enumerate(_poses_2d):
                inp_poses_2d[cnt] = _poses_2d[cam][b]
                poses_3d_tr[cnt] = _poses_3d[cam +'_3d_tr'][b]
                poses_3d_gt[cnt] = _poses_3d[cam +'_3d_gt'][b]
                inp_confs_2d[cnt] = sample['confidences'][c_idx][b]
                cam_ids[cnt] = c_idx
                cnt += 1
        '''
        Now we have data in the right format:

        poses_2d: (batch_size * num_cameras, 15*2)
        inp_confidences: (batch_size * num_cameras, 15)
        poses_3d_tr: (batch_size * num_cameras, 15, 3)
        poses_3d_gt: (batch_size * num_cameras, 15, 3)
        cam_ids: (batch_size * num_cameras)
        '''
        pred = None
        with torch.no_grad():
            pred = pose_lifter(inp_poses_2d, inp_confs_2d)
        pred_poses_3d = pred[0]
        pred_cam_angles = pred[1]
        
        # reproject to original cameras after applying rotation to the canonical poses and get triangulation error
        pred_rot = transform.euler_angles_to_matrix(pred_cam_angles, convention=['X','Y','Z'])

        # print(cam_ids.shape, pred_poses.shape, '|', out_poses_triang.shape, '|') #, traing_mpjpe_err.shape)

        out_cam_ids.append(cam_ids.cpu())
        out_pred_poses.append(pred_poses_3d.cpu())
        out_pred_rots.append(pred_rot.cpu())
        out_triang_poses.append(poses_3d_tr.cpu())
        out_gt_poses.append(poses_3d_gt.cpu())

        # Errors for checking
        rot_poses = torch.transpose(pred_rot.matmul(pred_poses_3d.reshape(-1, 3, num_joints)), 2, 1)
        # print("rot_poses: {}, poses_3d_tr: {}, poses_3d_gt: {}".format(rot_poses.shape, poses_3d_tr.shape, poses_3d_gt.shape))
        vanilla_mean_err = n_mpjpe(rot_poses, poses_3d_gt).unsqueeze(0) #* poses_3d_gt.shape[0]
        triangulated_mean_err = n_mpjpe(poses_3d_tr, poses_3d_gt).unsqueeze(0) #* poses_3d_gt.shape[0]

        vanilla_errs.append(vanilla_mean_err)
        triangulated_errs.append(triangulated_mean_err)

    # Print errors
    vanilla_errs = torch.cat(vanilla_errs) * 1000
    triangulated_errs = torch.cat(triangulated_errs) * 1000
    print("TRAIN Vanilla err: {:.6f}, triangulated err: {:.6f}".format(vanilla_errs.mean(), triangulated_errs.mean()))

    if config.save_outputs:
        # Output
        out_cam_ids = torch.cat(out_cam_ids, dim=0)
        out_pred_poses = torch.cat(out_pred_poses, dim=0)
        out_pred_rots = torch.cat(out_pred_rots, dim=0)
        out_triang_poses = torch.cat(out_triang_poses, dim=0)
        out_gt_poses = torch.cat(out_gt_poses, dim=0)
        out_dir = 'data/uncertnet/'
        np.save(out_dir + 'h36m_train_cam_ids.npy', out_cam_ids.numpy())
        np.save(out_dir + 'h36m_train_pred_poses.npy', out_pred_poses.numpy())
        np.save(out_dir + 'h36m_train_pred_rots.npy', out_pred_rots.numpy())
        np.save(out_dir + 'h36m_train_triang_poses.npy', out_triang_poses.numpy())
        np.save(out_dir + 'h36m_train_gt_poses.npy', out_gt_poses.numpy())

        # Load to check
        out_cam_ids = np.load(out_dir + 'h36m_train_cam_ids.npy')
        out_pred_poses = np.load(out_dir + 'h36m_train_pred_poses.npy')
        out_pred_rots = np.load(out_dir + 'h36m_train_pred_rots.npy')
        out_triang_poses = np.load(out_dir +'h36m_train_triang_poses.npy')
        out_gt_poses = np.load(out_dir +'h36m_train_gt_poses.npy')
        print("OUTPUTS: ")
        print(out_cam_ids.shape, out_pred_poses.shape, out_pred_rots.shape, out_triang_poses.shape, out_gt_poses.shape)


    print(" ---- TEST SET ----")
    out_cam_ids = []
    out_pred_poses = []
    out_pred_rots = []
    out_triang_poses = []
    out_gt_poses = []
    vanilla_errs = []
    triangulated_errs = []
    for i, sample in enumerate(tqdm(test_loader)):
        # print(sample.keys())
        _poses_2d = {key:sample[key] for key in config.all_cams}
        _poses_3d = {key:sample[key] for key in config.all_cams_3d}

        inp_poses_2d = torch.zeros((_poses_2d['cam0'].shape[0] * len(config.all_cams), num_joints*2)).cuda()
        poses_3d_tr = torch.zeros((_poses_3d['cam0_3d_tr'].shape[0] * len(config.all_cams), num_joints,3)).cuda()
        poses_3d_gt = torch.zeros((_poses_3d['cam0_3d_gt'].shape[0] * len(config.all_cams), num_joints,3)).cuda()
        inp_confs_2d = torch.zeros((_poses_2d['cam0'].shape[0] * len(config.all_cams), num_joints)).cuda()

        cam_ids = torch.zeros((_poses_2d['cam0'].shape[0] * len(config.all_cams))).cuda()

        cnt = 0
        for b in range(_poses_2d['cam0'].shape[0]):
            for c_idx, cam in enumerate(_poses_2d):
                inp_poses_2d[cnt] = _poses_2d[cam][b]
                poses_3d_tr[cnt] = _poses_3d[cam +'_3d_tr'][b]
                poses_3d_gt[cnt] = _poses_3d[cam +'_3d_gt'][b]
                inp_confs_2d[cnt] = sample['confidences'][c_idx][b]
                cam_ids[cnt] = c_idx
                cnt += 1
        '''
        Now we have data in the right format:

        poses_2d: (batch_size * num_cameras, 15*2)
        inp_confidences: (batch_size * num_cameras, 15)
        poses_3d_tr: (batch_size * num_cameras, 15, 3)
        poses_3d_gt: (batch_size * num_cameras, 15, 3)
        cam_ids: (batch_size * num_cameras)
        '''
        pred = None
        with torch.no_grad():
            pred = pose_lifter(inp_poses_2d, inp_confs_2d)
        pred_poses_3d = pred[0]
        pred_cam_angles = pred[1]
        
        # reproject to original cameras after applying rotation to the canonical poses and get triangulation error
        pred_rot = transform.euler_angles_to_matrix(pred_cam_angles, convention=['X','Y','Z'])

        # print(cam_ids.shape, pred_poses.shape, '|', out_poses_triang.shape, '|') #, traing_mpjpe_err.shape)

        out_cam_ids.append(cam_ids.cpu())
        out_pred_poses.append(pred_poses_3d.cpu())
        out_pred_rots.append(pred_rot.cpu())
        out_triang_poses.append(poses_3d_tr.cpu())
        out_gt_poses.append(poses_3d_gt.cpu())

        # Errors for checking
        rot_poses = torch.transpose(pred_rot.matmul(pred_poses_3d.reshape(-1, 3, num_joints)), 2, 1)
        # print("rot_poses: {}, poses_3d_tr: {}, poses_3d_gt: {}".format(rot_poses.shape, poses_3d_tr.shape, poses_3d_gt.shape))
        vanilla_mean_err = n_mpjpe(rot_poses, poses_3d_gt).unsqueeze(0) #* poses_3d_gt.shape[0]
        triangulated_mean_err = n_mpjpe(poses_3d_tr, poses_3d_gt).unsqueeze(0) #* poses_3d_gt.shape[0]

        vanilla_errs.append(vanilla_mean_err)
        triangulated_errs.append(triangulated_mean_err)

    # Print errors
    vanilla_errs = torch.cat(vanilla_errs) * 1000
    triangulated_errs = torch.cat(triangulated_errs) * 1000
    print("TEST Vanilla err: {:.6f}, triangulated err: {:.6f}".format(vanilla_errs.mean(), triangulated_errs.mean()))

    if config.save_outputs:
        # Output
        out_cam_ids = torch.cat(out_cam_ids, dim=0)
        out_pred_poses = torch.cat(out_pred_poses, dim=0)
        out_pred_rots = torch.cat(out_pred_rots, dim=0)
        out_triang_poses = torch.cat(out_triang_poses, dim=0)
        out_gt_poses = torch.cat(out_gt_poses, dim=0)
        out_dir = 'data/uncertnet/'
        np.save(out_dir + 'h36m_test_cam_ids.npy', out_cam_ids.numpy())
        np.save(out_dir + 'h36m_test_pred_poses.npy', out_pred_poses.numpy())
        np.save(out_dir + 'h36m_test_pred_rots.npy', out_pred_rots.numpy())
        np.save(out_dir + 'h36m_test_triang_poses.npy', out_triang_poses.numpy())
        np.save(out_dir + 'h36m_test_gt_poses.npy', out_gt_poses.numpy())

        # Load to check
        out_cam_ids = np.load(out_dir + 'h36m_test_cam_ids.npy')
        out_pred_poses = np.load(out_dir + 'h36m_test_pred_poses.npy')
        out_pred_rots = np.load(out_dir + 'h36m_test_pred_rots.npy')
        out_triang_poses = np.load(out_dir +'h36m_test_triang_poses.npy')
        out_gt_poses = np.load(out_dir +'h36m_test_gt_poses.npy')
        print("OUTPUTS: ")
        print(out_cam_ids.shape, out_pred_poses.shape, out_pred_rots.shape, out_triang_poses.shape, out_gt_poses.shape)
    
if __name__ == '__main__':
    main()

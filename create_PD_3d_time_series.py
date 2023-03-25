import argparse
import torch
import torch.nn
import torch.optim
from torch.utils import data
import numpy as np
import model_confidences
import os
from pytorch3d.transforms import so3_exponential_map as rodrigues
from utils.mohsen_utils.data_PD import *
from utils.pose_utils import procrustes_torch


def get_preds(input_args):

    backbone_2d_dataset_path = input_args.AP_data_path

    # --------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------

    # TODO: FINISH CHANGING THIS OVER FROM MOHSENS STRUCTURE TO MINE

    keypoints_PD = np.load(backbone_2d_dataset_path, allow_pickle=True)
    keypoints_PD = keypoints_PD['positions_2d'].item()
    num_joints = 15 

    for subject in keypoints_PD.keys():
        for action in keypoints_PD[subject]:
            for cam_idx in range(len(keypoints_PD[subject][action]['pos'])):
                # Normalize camera frame
                kps = keypoints_PD[subject][action]['pos'][cam_idx]
                conf = keypoints_PD[subject][action]['conf'][cam_idx]
                kps = kps[:, [0,1,2,3,4,5,6,8,9,10,11,12,13,14,15], :]
                conf = conf[:, [0,1,2,3,4,5,6,8,9,10,11,12,13,14,15], :]
                kps = kps - kps[:, :1, :]
                kps = np.transpose(kps, [0, 2, 1])
                kps = kps.reshape(-1, num_joints*2)
                kps /= np.linalg.norm(kps, ord=2, axis=1, keepdims=True)
                keypoints_PD[subject][action]['pos'][cam_idx] = kps
                keypoints_PD[subject][action]['conf'][cam_idx] = conf

    def fetch_train(subjects, actions):
        out_subject = []
        out_poses_2d = []
        out_confidences=[]
        for i in range(2):
            out_poses_2d.append([])
            out_confidences.append([])

        for subject in subjects:
            for action in actions:
                    
                poses_2d = keypoints_PD[subject][action]['pos']
                conf_2d = keypoints_PD[subject][action]['conf']

                for i in range(2): # Iterate across cameras
                    out_poses_2d[i].append(poses_2d[i])
                    out_confidences[i].append(conf_2d[i])

                out_subject.append(np.ones(len(poses_2d[0])) * int(subject[1:]))
                
        for i in range(len(poses_2d)): # Iterate across cameras
            out_poses_2d[i] = np.concatenate(out_poses_2d[i], axis=0)        
            out_confidences[i] = np.concatenate(out_confidences[i], axis=0) 
        out_subject = np.concatenate(out_subject, axis=0)

        return out_poses_2d, out_confidences, out_subject

    subjects = [['S01'],['S02'],['S03'],['S04'],['S05'],['S06'],
                ['S07'],['S08'],['S09'],['S10'],['S11'],
                ['S12'],['S13'],['S14'],['S15'],['S16'],
                ['S17'],['S18'],['S19'],['S20'],['S21'],
                ['S22'],['S23'],['S24'],['S25'],['S26'],
                ['S27'],['S28'],['S29'],['S30'],['S31'],
                ['S32'],['S33'],['S34'],['S35']]

    cam_names = ['54138969', '55011271', '58860488', '60457274']
    all_cams = ['cam0', 'cam1']
    cam_ch_names = ['CH3', 'CH4']

    # --------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------


    # Options
    outpath = input_args.outpath
    lifter_model_path = input_args.model_ckpt_path
    camspace_out = input_args.camspace_out

    print('camspace_reproj: ', camspace_out)

    # loading the lifting network
    model_eval = model_confidences.Lifter().cuda()

    # Get the predictions
    with torch.no_grad():
        checkpoint = torch.load(lifter_model_path)
        model_eval.load_state_dict(checkpoint.state_dict())
        model_eval.eval()

        for cam_name_idx, cam in enumerate(all_cams):
            # Output paths
            preds_outpath = (outpath + cam_ch_names[cam_name_idx] + '/finetune_3d_camspace/') if camspace_out else \
                            (outpath + cam_ch_names[cam_name_idx] + '/finetune_3d_canonspace/')

            if not os.path.exists(preds_outpath):
                print("Creating output directory: ", preds_outpath)
                os.makedirs(preds_outpath)

            print("Saving predictions to: ", preds_outpath)
            preds_outpath += 'Predictions_'

            for subj in subjects:
                pred_save = []
                poses_2d_train2, conf_2d_train2, out_subject2 = fetch_train(subj,['WalkingOval'])
                my_dataset2 = H36MDataset(poses_2d_train2, conf_2d_train2, out_subject2, normalize_2d=True)
                train_loader2 = data.DataLoader(my_dataset2, batch_size=2000, shuffle=False, num_workers=0)    

                for i, sample in enumerate(train_loader2):
                    # not the most elegant way to extract the dictionary
                    poses_2d = {key:sample[key] for key in all_cams}

                    inp_poses = torch.zeros((poses_2d[cam].shape[0] , num_joints*2)).cuda()
                    inp_confidences = torch.zeros((poses_2d[cam].shape[0] , num_joints)).cuda()

                    # poses_2d is a dictionary. It needs to be reshaped to be propagated through the model.
                    cnt = 0
                    for b in range(poses_2d[cam].shape[0]):
                        inp_poses[cnt] = poses_2d[cam][b]
                        inp_confidences[cnt] = sample['confidences'][0][b]
                        cnt += 1

                    pred = model_eval(inp_poses, inp_confidences)
                    pred_poses = pred[0]
                    pred_cam_angles = pred[1]

                    # Reproject to camera if we want
                    if camspace_out:
                        # angles are in axis angle notation, use Rodrigues formula (Equations 3 and 4) to get the rotation matrix
                        pred_rot = rodrigues(pred_cam_angles)
                        # reproject to original cameras after applying rotation to the canonical poses
                        pred_poses = pred_rot.matmul(pred_poses.reshape(-1, 3, num_joints))
                    else:
                        pred_poses = pred_poses.reshape(-1, 3, 15)    # (B, XYZ, 15)
                    
                    out_preds = np.transpose(pred_poses.cpu(), [0, 2, 1])

                    # Zero preds to hip and align to first frame if we want
                    if input_args.align_preds:
                        pred_poses -= pred_poses[:,:1]
                        out_preds = procrustes_torch(pred_poses[0:1], pred_poses)

                    pred_save.append(out_preds)
                
                print(subj[0], end=', ')
                # print(pred_save[0][0,:1], end=', ')
                print(pred_save[0].shape)
                np.save(preds_outpath + subj[0], np.concatenate(pred_save))
        print(' done!')


def get_args():
    parser = argparse.ArgumentParser()
    # Tasks
    parser.add_argument("--train", default=False, help="train model?", type=bool)
    parser.add_argument("--get_preds", default=True, help="use model to get 3D prediction timeseries?", type=bool)
    # Data
    parser.add_argument("--AP_data_path", default="data/body/2d_proposals/mohsen_data_PD.npz", help="path to alphapose 2d preds data")
    parser.add_argument("--model_ckpt_path", default="model_checkpoints/body_pose/Mohsens/model_pretrain.pt", help="path to model checkpoint")
    # Output
    parser.add_argument("--outpath", default="data/body/3d_time_series/", help="path to save predictions")
    # Output Options
    parser.add_argument("--camspace_out", default=False, help="transform outputs from canonical to camera space?", type=bool)
    parser.add_argument("--align_preds", default=False, help="zero to hip and do procrustes on preds?", type=bool)
    return parser.parse_args()

def main():
    input_args = get_args()
    # Do the tasks we want
    if input_args.get_preds:
        get_preds(input_args)
    if input_args.train:
        raise NotImplementedError

if __name__ == '__main__':
    main()
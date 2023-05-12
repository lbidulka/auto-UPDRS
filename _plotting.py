import torch
import cv2
import argparse
from utils import info, post_processing, helpers, pose_visualization, metrics, pose_utils
from sklearn.metrics import accuracy_score
import numpy as np
import data.body.body_dataset as body_dataset
from data.body.body_dataset import body_ts_loader
import models.body_pose as body_nets
import pickle
from tqdm import tqdm

def plot_thing(pred_3d, in_frames, out_fig_path):
    '''
    plot 3d knee coord as x, and frame idx of inference channel as y

    args:
        pred_3d: (n_frames, 3, 17)
        in_frames: (n_frames, )
    '''
    import matplotlib.pyplot as plt

    fig = plt.figure()
    spec = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(spec[0, 0])

    # get LL and RL Hip->LKnee vectors
    LL_hip_knee = pred_3d[:, :, info.PD_3D_skeleton_kpt_idxs['LL']['LKnee']] - pred_3d[:, :, info.PD_3D_skeleton_kpt_idxs['LL']['Hip']]
    RL_hip_knee = pred_3d[:, :, info.PD_3D_skeleton_kpt_idxs['RL']['RKnee']] - pred_3d[:, :, info.PD_3D_skeleton_kpt_idxs['RL']['Hip']]

    # get T Neck->Hip vector
    T_neck_hip = pred_3d[:, :, info.PD_3D_skeleton_kpt_idxs['T']['Neck']] - pred_3d[:, :, info.PD_3D_skeleton_kpt_idxs['T']['Hip']]

    # avg LL and RL Hip->LKnee vectors, and get angle between that and T Neck->Hip vector
    LL_RL_hip_knee = (LL_hip_knee + RL_hip_knee) / 2
    LL_RL_hip_knee = LL_RL_hip_knee / np.linalg.norm(LL_RL_hip_knee, axis=1, keepdims=True)
    T_neck_hip = T_neck_hip / np.linalg.norm(T_neck_hip, axis=1, keepdims=True)
    T_to_thigh_angle = np.arccos(np.sum(LL_RL_hip_knee * T_neck_hip, axis=1))

    # plot
    ax.scatter(in_frames, T_to_thigh_angle, c='b', s=0.2, label='pred',)
    ax.set_title('Some val over time')
    ax.set_xlabel('timestep (img frame idx)')
    ax.set_ylabel('val')    

    plt.savefig(out_fig_path, dpi=500, bbox_inches='tight')
    plt.close(fig)

# Full body tracking tasks (e.g. gait analysis)
def body_tasks(input_args):

    task = 'tug_stand_walk_sit'
    # task = 'free_form_oval'
    chs = ['002', '006']
    inf_ch_idx = 1

    dataset_path = './auto_UPDRS/data/body/'

    body_2d_kpts_path = "{}2d_proposals/all_tasks_2D_kpts.pickle".format(dataset_path,)
    body_3d_preds_path = "{}3d_time_series/all_tasks_CH{}.pickle".format(dataset_path, chs[inf_ch_idx])
    # body_2d_kpts_path = "{}2d_proposals/{}_CH{}_CH{}_2D_kpts.pickle".format(dataset_path, task, chs[0], chs[1])
    # body_3d_preds_path = "{}3d_time_series/CH{}_{}.pickle".format(dataset_path, chs[inf_ch_idx], task)

    # ts_loader = body_ts_loader(body_3d_preds_path, task, subjects=info.subjects_All, pickled=True)
    # ts_loader = body_ts_loader(body_3d_preds_path, task, subjects=info.subjects_All, pickled=True)
    ts_loader = body_ts_loader(body_3d_preds_path, task, subjects=[subj for subj in info.subjects_All if subj not in ['S21']], # S21 is too short
                               pickled=True, proc_aligned=False, zero_rot=True, smoothing=False)

    for S_id in ['S01']:
        # load 3d pickle
        with open(body_3d_preds_path, 'rb') as f:
            body_3d_preds = pickle.load(f)
        pred_3d_kpts = np.transpose(ts_loader.get_data_norm(S_id), (0, 2, 1))

        data_2d = body_dataset.get_2D_data([S_id], [task], body_2d_kpts_path, normalized=True, old_sys_return_only=inf_ch_idx)
        (ch0_2d_kpts, ch1_2d_kpts, ch0_2d_confs, ch1_2d_confs, ch0_frames, ch1_frames) = data_2d

        # plot 3d knee coord as x, and frame idx of inference channel as y
        plot_thing(pred_3d_kpts, (ch1_frames if inf_ch_idx else ch0_frames), out_fig_path="./auto_UPDRS/outputs/pose.png")

        # plot pred 3d pose and 2d input poses
        # for i in tqdm(range(70, ch0_2d_kpts.shape[0])):
        # for i in tqdm(range(ch0_2d_kpts.shape[0])):
        #     pose_visualization.visualize_multi_view_pose(pred_3d_kpts[i], kpts_2D=[ch0_2d_kpts[i], ch1_2d_kpts[i]], 
        #                                                  lifter_in_view=inf_ch_idx, 
        #                                                  save_fig=True, out_fig_path="./auto_UPDRS/outputs/pose.png")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_path", default="./auto_UPDRS/model_checkpoints/", help="model checkpoints path", type=str)
    parser.add_argument("--data_path", default="/mnt/CAMERA-data/CAMERA/Other/lbidulka_dataset/", help="input data path", type=str)
    parser.add_argument("--output_path", default="./auto_UPDRS/outputs/", help="output data path", type=str)
    return parser.parse_args()

def main():
    input_args = get_args()
    body_tasks(input_args)    
    
if __name__ == '__main__':
    main()

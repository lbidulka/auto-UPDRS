import torch
import cv2
import argparse
from utils import info, post_processing, helpers, pose_visualization, metrics, pose_utils
import numpy as np
import data.body.body_dataset as body_dataset
from data.body.body_dataset import body_ts_loader
import models.body_pose as body_nets
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy.signal import argrelextrema


def detect_TUG_action(S_id, pred_3d, in_frames,):
    '''
    Classifies each of the timeseries frames as 
        0: 'sit'
        1: 'walk'
    '''

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def plot_thing(S_id, pred_3d, in_frames, out_fig_path):
    '''
    plot 3d knee coord as x, and frame idx of inference channel as y

    args:
        S_id: str
        pred_3d: (n_frames, 3, 17)
        in_frames: (n_frames, )
    '''
    fig = plt.figure()
    spec = fig.add_gridspec(1, 1)
    ax1 = fig.add_subplot(spec[0, 0])
    ax2 = ax1.twinx() 

    # Get angle and smooth, then plot
    T_to_thigh_angle = post_processing.get_T_thigh_angle(pred_3d)
    n = 10
    # T_to_thigh_angle = moving_average(T_to_thigh_angle, n=n)

    ax1.scatter(in_frames, T_to_thigh_angle, c='b', s=0.2, label='ang',)
    # ax1.scatter(in_frames[n-1:], T_to_thigh_angle, c='b', s=0.2, label='ang',)
    ax1.set_title('{} Torso-Thigh angle'.format(S_id))
    ax1.set_xlabel('timestep (img frame idx)')
    ax1.set_ylabel('angle (rad)')


    # # Get extrema, then plot
    extrema_ord = 15
    maxima_idx = argrelextrema(T_to_thigh_angle, np.greater, order=extrema_ord)
    minima_idx = argrelextrema(T_to_thigh_angle, np.less, order=extrema_ord)

    maxima = T_to_thigh_angle[maxima_idx]
    minima = T_to_thigh_angle[minima_idx]

    # ax1.scatter(in_frames[n-1:][maxima_idx], maxima, c='g', s=10, label='max', marker='^')
    # ax1.scatter(in_frames[n-1:][minima_idx], minima, c='m', s=10, label='min', marker='v')
    ax1.scatter(in_frames[maxima_idx], maxima, c='g', s=10, label='max', marker='^')
    ax1.scatter(in_frames[minima_idx], minima, c='m', s=10, label='min', marker='v')

    # threshold the angle to classify actions
    # n = 10
    # ang_walk_thresh = 2.7
    # action_class = moving_average(T_to_thigh_angle, n=n) > ang_walk_thresh
    # action_class = action_class.astype(int)
    # action_class = np.concatenate((np.zeros(n-1), action_class))
    action_class = post_processing.classify_tug(pred_3d)

    ax2.scatter(in_frames, action_class, c='r', s=0.2, label='action',)



    # Get diffs btw maximas and minimas, then plot
    # maxima_diff = np.diff(maxima)
    # minima_diff = np.diff(minima)

    # ax1.scatter(in_frames[n-1:][maxima_idx][1:], maxima_diff, c='c', s=10, label='d_max', marker='2')
    # ax1.scatter(in_frames[n-1:][minima_idx][1:], minima_diff, c='r', s=10, label='d_min', marker='3')


    # compute discrete derivative of T_to_thigh_angle
    # T_to_thigh_angle_diff = np.diff(T_to_thigh_angle)

    # ax2.scatter(in_frames[n:], T_to_thigh_angle_diff, c='r', s=0.2, label='d_ang',)

    # Legend and show
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2)

    plt.savefig(out_fig_path, dpi=500, bbox_inches='tight')
    plt.close(fig)

# Full body tracking tasks (e.g. gait analysis)
def body_tasks():

    task = 'tug_stand_walk_sit'
    # task = 'free_form_oval'
    chs = ['002', '006']
    inf_ch_idx = 1

    dataset_path = './auto_UPDRS/data/body/'
    body_2d_kpts_path = "{}2d_proposals/all_tasks_2D_kpts.pickle".format(dataset_path,)
    body_3d_preds_path = "{}3d_time_series/all_tasks_CH{}.pickle".format(dataset_path, chs[inf_ch_idx])

    ts_loader = body_ts_loader(body_3d_preds_path, body_2d_kpts_path, task, subjects=[subj for subj in info.subjects_All if subj not in ['S21']], # TUG S21 is too short
                                pickled=True, proc_aligned=False, zero_rot=True, smoothing=False)

    # Feature extraction
    print("Extracting features...")
    feature_processor = post_processing.gait_processor(ts_loader, "./auto_UPDRS/outputs/")
    print("Plotting features...")
    feature_processor.plot_feats_ts(show=False)

    # Single Subj plotting
    for S_id in ['S01']:
        # load 3d pickle
        # with open(body_3d_preds_path, 'rb') as f:
        #     body_3d_preds = pickle.load(f)

        pred_3d_kpts = np.transpose(ts_loader.get_data_norm(S_id), (0, 2, 1))

        data_2d = body_dataset.get_2D_data([S_id], [task], body_2d_kpts_path, normalized=True, old_sys_return_only=inf_ch_idx)
        (ch0_2d_kpts, ch1_2d_kpts, ch0_2d_confs, ch1_2d_confs, ch0_frames, ch1_frames) = data_2d

        # plot torso-thing angle for timing
        plot_thing(S_id, pred_3d_kpts, (ch1_frames if inf_ch_idx else ch0_frames), out_fig_path="./auto_UPDRS/outputs/pose.png")       

        # plot pred poses
        # for i in tqdm(range(223, ch0_2d_kpts.shape[0])):
        # for i in tqdm(range(ch0_2d_kpts.shape[0])):
            # pose_visualization.visualize_multi_view_pose(pred_3d_kpts[i], kpts_2D=[ch0_2d_kpts[i], ch1_2d_kpts[i]], 
            #                                              lifter_in_view=inf_ch_idx, 
            #                                              save_fig=True, out_fig_path="./auto_UPDRS/outputs/pose.png")

def main():
    body_tasks()
    
if __name__ == '__main__':
    main()

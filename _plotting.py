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
import time

from sklearn.metrics import accuracy_score
import scipy.signal
from scipy.signal import argrelextrema


def _moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

def plot_xcorr(x, y, out_fig_path): 
    '''
    Plot cross-correlation (full) between two signals.
    '''
    N = max(len(x), len(y)) 
    n = min(len(x), len(y)) 

    if N == len(y): 
        lags = np.arange(-N + 1, n) 
    else: 
        lags = np.arange(-n + 1, N) 
    c = scipy.signal.correlate(x / np.std(x), y / np.std(y), 'full') 

    fig = plt.figure()
    plt.plot(lags, c / n) 
    plt.savefig(out_fig_path, dpi=500, bbox_inches='tight')
    plt.close(fig)

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

    ax1.scatter(in_frames, T_to_thigh_angle, c='b', s=0.2, label='ang',)
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


    # Get diffs btw maximas and minimas, then plot
    # maxima_diff = np.diff(maxima)
    # minima_diff = np.diff(minima)

    # ax1.scatter(in_frames[n-1:][maxima_idx][1:], maxima_diff, c='c', s=10, label='d_max', marker='2')
    # ax1.scatter(in_frames[n-1:][minima_idx][1:], minima_diff, c='r', s=10, label='d_min', marker='3')


    # compute discrete derivative of T_to_thigh_angle
    diff_smooth_n = 15
    T_to_thigh_angle_diff = post_processing.get_d_T_thigh_angle(T_to_thigh_angle, n=diff_smooth_n)

    ax2.scatter(in_frames[diff_smooth_n:], T_to_thigh_angle_diff, c='c', s=0.2, label='d_ang',)
    # ax2.scatter(in_frames, T_to_thigh_angle_diff, c='c', s=0.2, label='d_ang',)

    # Get and plot action class
    action_class, abs_transition = post_processing.classify_tug(pred_3d, n=diff_smooth_n)

    ax2.scatter(in_frames, action_class, c='r', s=0.2, label='action',)
    ax2.scatter(in_frames, abs_transition, c='y', s=0.2,)

    # Legend and show
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2)

    plt.savefig(out_fig_path, dpi=500, bbox_inches='tight')
    plt.close(fig)


def naive_voting(gait_processor, subjs):
    # Use thresholds to get predictions for PD subjects by majority voting
    pd_indices = [i for i, x in enumerate(subjs) if (x in info.subjects_PD)]
    pd_gait_feats = gait_processor.feats_avg[:, pd_indices][:-1]
    pd_indicators = gait_processor.compute_indicators(pd_gait_feats)
    pd_indicators_grouped = gait_processor.compute_indicators(pd_gait_feats, grouped=True)


    # Results
    label_indices = [i for i, x in enumerate(info.subjects_PD) if (x in subjs)]
    labels = info.Y_true[label_indices]

    print("\nlabels:\n", labels)
    # gait_processor.plot_feats(subjs, thresholds=True)
    # gait_processor.plot_preds_by_feats(pd_gait_feats, pd_indicators)

    # get accuracy for each feature
    feat_accs = [np.round(accuracy_score(labels, pd_indicators[i]), 4) for i in range(len(info.clinical_gait_feat_names))]
    group_feat_accs = [np.round(accuracy_score(labels, pd_indicators_grouped[i]), 4) for i in range(8)]

    print("\n individ. feat accs:\n", info.clinical_gait_feat_acronyms, "\n", feat_accs)
    print("\n group feat accs:\n", info.clinical_gait_feat_acronyms_group, "\n", group_feat_accs)

    print("\n----------- >= n voting COMBOS -----------")
    ns = [1, 2, 3, 4]
    for n in ns:
        print("n: ", n)
        # 0. all
        preds = np.sum(pd_indicators, axis=0) >= n
        acc = accuracy_score(labels, np.round(preds))
        print("  all_acc: ", np.round(acc, 4))
        # 0. all, grouped
        preds = np.sum(pd_indicators_grouped, axis=0) >= n
        acc = accuracy_score(labels, np.round(preds))
        print("  all_acc_grouped: ", np.round(acc, 4))
        # 1. RSL, LSL, RFC, LFC
        preds = np.sum(pd_indicators[[1,2,4,5]], axis=0) >= n
        acc = accuracy_score(labels, np.round(preds))
        print("  RSL_LSL_RFC_LFC acc: ", np.round(acc, 4))

        # # 2. RSL, LSL
        # preds = np.sum(pd_indicators[[1,2]], axis=0) >= n
        # acc = accuracy_score(labels, np.round(preds))
        # print("  RSL_LSL_acc: ", np.round(acc, 4))
        # # 3. RSL, LSL, RKF, LKF
        # preds = np.sum(pd_indicators[[1,2,4,5]], axis=0) >= n
        # acc = accuracy_score(labels, np.round(preds))
        # print("  RSL_LSL_RKF_LKF_acc: ", np.round(acc, 4))


# Full body tracking tasks (e.g. gait analysis)
def body_tasks():

    task = 'tug_stand_walk_sit'
    # task = 'free_form_oval'
    # task = 'arising_chair'
    chs = ['002', '006']
    inf_ch_idx = 1

    subjs = [subj for subj in info.subjects_All if subj not in ['S21']] # TUG S21 is too short
    # subjs = info.subjects_new_sys

    dataset_path = './auto_UPDRS/data/body/'
    body_2d_kpts_path = "{}2d_proposals/all_tasks_2D_kpts.pickle".format(dataset_path,)
    body_3d_preds_path = "{}3d_time_series/all_tasks_CH{}.pickle".format(dataset_path, chs[inf_ch_idx])

    ts_loader = body_ts_loader(body_3d_preds_path, body_2d_kpts_path, task, subjects=subjs,
                                pickled=True, proc_aligned=False, zero_rot=True, smoothing=False)

    # Feature extraction
    # print("Extracting features...")
    # feature_processor = post_processing.gait_processor(ts_loader, "./auto_UPDRS/outputs/")
    # print("Plotting features...")
    # feature_processor.plot_feats_ts(show=False)

    # Simple voting classifiers
    # naive_voting(feature_processor, subjs)

    # Single Subj plotting
    for S_id in ['S16']:
    # for S_id in [subj for subj in info.subjects_All if subj not in ['S21']]:
        # load 3d pickle
        # with open(body_3d_preds_path, 'rb') as f:
        #     body_3d_preds = pickle.load(f)

        pred_3d_kpts = np.transpose(ts_loader.get_data_norm(S_id), (0, 2, 1))

        data_2d = body_dataset.get_2D_data([S_id], [task], body_2d_kpts_path, normalized=True, old_sys_return_only=inf_ch_idx)
        (ch0_2d_kpts, ch1_2d_kpts, ch0_2d_confs, ch1_2d_confs, ch0_frames, ch1_frames) = data_2d

        # plot torso-thing angle for timing
        plot_thing(S_id, pred_3d_kpts, (ch1_frames if inf_ch_idx else ch0_frames), out_fig_path="./auto_UPDRS/outputs/pose.png")

        time.sleep(1)

        # plot pred poses
        # for i in tqdm(range(225, ch0_2d_kpts.shape[0])):
        # # for i in tqdm(range(125, ch0_2d_kpts.shape[0])):
        # # for i in tqdm(range(ch0_2d_kpts.shape[0])):
        #     pose_visualization.visualize_multi_view_pose(pred_3d_kpts[i], kpts_2D=[ch0_2d_kpts[i], ch1_2d_kpts[i]], 
        #                                                  lifter_in_view=inf_ch_idx, 
        #                                                  save_fig=True, out_fig_path="./auto_UPDRS/outputs/pose.png")

def main():
    body_tasks()
    
if __name__ == '__main__':
    main()

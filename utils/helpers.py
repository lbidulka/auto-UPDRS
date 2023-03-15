import torch
import numpy as np
import models.body_pose as body_nets
from utils import info, post_processing, pose_visualization


def make_vids(dims=3, mohsens_preds=False, subjs=None, outdir="outputs/"):
    '''
    Quick function to create some videos from input pose data
    '''    
    if dims == 2:
        in_file = "outputs/vids_2d/2D_kpts.npy"
        pred_aligned = np.load(in_file)
        pose_visualization.pose2d_video(pred_aligned, outpath="outputs/vids_2d/")

    elif dims == 3:
        if mohsens_preds:
            for subj in subjs:
                print("Processing: ", subj)
                in_file = outdir + "../data/body/time_series/outputs_finetuned/Predictions_" + subj + ".npy"
                outpath = outdir + "vids_3d/mohsens_preds/"
                pred_aligned = np.load(in_file)
                pred_aligned = np.swapaxes(pred_aligned, 1, 2)[:150]    # TODO: REMOVE THIS HARDCODED SLICE
                pose_visualization.pose3d_video(pred_aligned, outpath=outpath, vid_name=subj + "_pose_3d.mp4")
        else:
            subj = '9769'   # TODO: REPLACE THIS WITH MULTI-SUBJECT LOOP
            in_file = outdir + "vids_3d/3D_kpts.npy"
            outpath = outdir + "vids_3d/"
            pred_aligned = np.load(in_file)
            pose_visualization.pose3d_video(pred_aligned, outpath=outpath, vid_name=subj + "_pose_3d.mp4")

def fix_model_setup(in_ckpt_path, out_dict_path):
    '''
    Helper to fix the model setup
    '''
    # load the pretrained model in Mohsens setup
    model = body_nets.Lifter()    # CHANGE THIS AS NECESSARY
    dict = torch.load(in_ckpt_path).state_dict()
    model.load_state_dict(dict)
    # Fix the setup by only saving the state_dict
    torch.save(model.state_dict(), out_dict_path)
    print("Saved new", out_dict_path)

def naive_voting(gait_processor):
    '''
    Some exploration of the gait processing and indicator functions
    '''
    # Use thresholds to get predictions for PD subjects by majority voting
    pd_indices = [i for i, x in enumerate(info.subjects_All) if x in info.subjects_PD]
    pd_gait_feats = gait_processor.feats_avg[:, pd_indices][:15]
    pd_indicators = gait_processor.compute_indicators(pd_gait_feats)
    pd_indicators_grouped = gait_processor.compute_indicators(pd_gait_feats, grouped=True)

    # Results
    print("\nlabels:\n", info.Y_true)
    gait_processor.plot_feats(info.subjects_All, thresholds=True)
    gait_processor.plot_preds_by_feats(pd_gait_feats, pd_indicators)

    # get accuracy for each feature
    # feat_accs = [np.round(accuracy_score(info.Y_true, pd_indicators[i]), 4) for i in range(len(info.clinical_gait_feat_names))]
    # print("\n individ. feat accs:\n", info.clinical_gait_feat_acronyms, "\n", feat_accs)
    # group_feat_accs = [np.round(accuracy_score(info.Y_true, pd_indicators_grouped[i]), 4) for i in range(8)]
    # print("\n group feat accs:\n", info.clinical_gait_feat_acronyms_group, "\n", group_feat_accs)

    # # print("\n----------- >= n voting COMBOS -----------")
    # ns = [1, 2, 3, 4]
    # for n in ns:
    #     print("n: ", n)
    #     # 0. all
    #     preds = np.sum(pd_indicators, axis=0) >= n
    #     acc = accuracy_score(info.Y_true, np.round(preds))
    #     print("  all_acc: ", np.round(acc, 4))
    #     # 0. all, grouped
    #     preds = np.sum(pd_indicators_grouped, axis=0) >= n
    #     acc = accuracy_score(info.Y_true, np.round(preds))
    #     print("  all_acc_grouped: ", np.round(acc, 4))
    #     # 1. RSL, LSL, Cad, RFH, LFH
    #     preds = np.sum(pd_indicators[[1,2,3,8,9]], axis=0) >= n
    #     acc = accuracy_score(info.Y_true, np.round(preds))
    #     print("  RSL_LSL_Cad_RFH_LFH_acc: ", np.round(acc, 4))
    #     # 2. RSL, LSL
    #     preds = np.sum(pd_indicators[[1,2]], axis=0) >= n
    #     acc = accuracy_score(info.Y_true, np.round(preds))
    #     print("  RSL_LSL_acc: ", np.round(acc, 4))
    #     # 3. RSL, LSL, RKF, LKF
    #     preds = np.sum(pd_indicators[[1,2,4,5]], axis=0) >= n
    #     acc = accuracy_score(info.Y_true, np.round(preds))
    #     print("  RSL_LSL_RKF_LKF_acc: ", np.round(acc, 4))
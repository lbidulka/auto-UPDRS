import torch
import models.body_pose as body_nets
from utils import info, post_processing

# Helper to fix the model setup
def fix_model_setup(in_ckpt_path, out_dict_path):
    # load the pretrained model in Mohsens setup
    model = body_nets.Lifter()    # CHANGE THIS AS NECESSARY
    dict = torch.load(in_ckpt_path).state_dict()
    model.load_state_dict(dict)
    # Fix the setup by only saving the state_dict
    torch.save(model.state_dict(), out_dict_path)
    print("Saved new", out_dict_path)

# Some exploration of the gait processing and indicator functions
def naive_voting(gait_processor):
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
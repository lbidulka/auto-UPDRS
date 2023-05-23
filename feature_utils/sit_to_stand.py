import numpy as np
from scipy.signal import savgol_filter, find_peaks
from utils import info, cam_sys_info


def get_elbow_angles(pred_3d):
    '''
    '''
    # get Elbow->Wrist and Elbow->Shoulder vectors
    L_elbow_wrist = pred_3d[:, :, info.PD_3D_skeleton_kpt_idxs['LElbow']] - pred_3d[:, :, info.PD_3D_skeleton_kpt_idxs['LWrist']]
    R_elbow_wrist = pred_3d[:, :, info.PD_3D_skeleton_kpt_idxs['RElbow']] - pred_3d[:, :, info.PD_3D_skeleton_kpt_idxs['RWrist']]

    L_elbow_shoulder = pred_3d[:, :, info.PD_3D_skeleton_kpt_idxs['LElbow']] - pred_3d[:, :, info.PD_3D_skeleton_kpt_idxs['LShoulder']]
    R_elbow_shoulder = pred_3d[:, :, info.PD_3D_skeleton_kpt_idxs['RElbow']] - pred_3d[:, :, info.PD_3D_skeleton_kpt_idxs['RShoulder']]

    # get angle between those vectors
    L_elbow_angle = np.arccos(np.sum(L_elbow_wrist * L_elbow_shoulder, axis=1))
    R_elbow_angle = np.arccos(np.sum(R_elbow_wrist * R_elbow_shoulder, axis=1))

    return (L_elbow_angle + R_elbow_angle) / 2


def get_hip_forearm_angle(pred_3d):
    '''
    '''
    # LHip->RHip and Elbow->Wrist vectors
    L_elbow_wrist = pred_3d[:, :, info.PD_3D_skeleton_kpt_idxs['LElbow']] - pred_3d[:, :, info.PD_3D_skeleton_kpt_idxs['LWrist']]
    R_elbow_wrist = pred_3d[:, :, info.PD_3D_skeleton_kpt_idxs['RElbow']] - pred_3d[:, :, info.PD_3D_skeleton_kpt_idxs['RWrist']]
    LHip_RHip = pred_3d[:, :, info.PD_3D_skeleton_kpt_idxs['LHip']] - pred_3d[:, :, info.PD_3D_skeleton_kpt_idxs['RHip']]

    # get angle between those vectors
    L_elbow_angle = np.arccos(np.sum(L_elbow_wrist * LHip_RHip, axis=1))
    R_elbow_angle = np.arccos(np.sum(R_elbow_wrist * LHip_RHip, axis=1))

    return (L_elbow_angle + R_elbow_angle) / 2

def get_hands_to_hip_dist(pred_3d):
    '''
    '''
    # Hand->Hip vectors
    L_hand_hip = pred_3d[:, :, info.PD_3D_skeleton_kpt_idxs['LWrist']] - pred_3d[:, :, info.PD_3D_skeleton_kpt_idxs['LHip']]
    R_hand_hip = pred_3d[:, :, info.PD_3D_skeleton_kpt_idxs['RWrist']] - pred_3d[:, :, info.PD_3D_skeleton_kpt_idxs['RHip']]

    return (np.linalg.norm(L_hand_hip, axis=1) + np.linalg.norm(R_hand_hip, axis=1)) / 2


def uses_hands_to_rise(pred_3d, action_class):
    '''
    detects if the subject uses their hands to rise from the chair

    args:
        pose_3d: (n_frames, 3, 17)      3D pose timeseries of subject
        action_class: (n_frames, )      sitting, transitioning, or walking for each frame?
    '''

    elbow_angs = get_elbow_angles(pred_3d)






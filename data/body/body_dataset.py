import torch
from torch.utils.data import Dataset
from scipy.signal import savgol_filter
import numpy as np
import json
import os

from utils import info, alphapose_filtering, cam_sys_info

def get_2D_data(subjs, task, data_path, normalized=True, mohsens_data=False, mohsens_output=False):
    '''
    Fetches the 2D pose data for specified subjects.

    TODO: ADD SUPPORT FOR ARBITRARY CHANNELS

    Args:
        subj (str): subject ID
        task (str): task name (e.g. 'free_form_oval', ...)
            chs (list): list of channels to fill (e.g. ['001', '002',])    NOTE: THIS USES NEW SYS NAMING
        data_path (str): path to the 2d kpts dataset
        normalize (bool): if True, normalize the data
        mohsens_data (bool): if True, load differently for Mohsens data, else use as is
        mohsens_output (bool): if True, return the data in the same format as mohsens data
    Returns: 
        ch3_data (np.array): 2D pose data for view 0
        ch4_data (np.array): 2D pose data for view 1
    '''
    # Load up all subjs
    keypoints_PD = np.load(data_path, allow_pickle=True)
    if mohsens_data:
        task = 'WalkingOval'
        keypoints_PD = keypoints_PD['positions_2d'].item()  # contains all subjects

    for subject in keypoints_PD.keys():
        for action in keypoints_PD[subject]:
            # keypoints_PD[subject][action]['subject'] = []
            for cam_idx in range(len(keypoints_PD[subject][action]['pos'])):
                kps = keypoints_PD[subject][action]['pos'][cam_idx]     # (n_frames, n_kpts, xy)
                conf = keypoints_PD[subject][action]['conf'][cam_idx]

                kps = kps[:,[0,1,2,3,4,5,6,8,9,10,11,12,13,14,15],:]
                conf = conf[:,[0,1,2,3,4,5,6,8,9,10,11,12,13,14,15],:]

                if normalized:
                    kps = kps - kps[:,:1,:]  # subtract root joint

                kps = np.transpose(kps, [0,2,1])
                kps = kps.reshape(-1, 15*2)  # flatten: (1, xxx ... yyy)

                if normalized:
                    kps /= np.linalg.norm(kps, axis=1, keepdims=True)  # normalize scale
                
                keypoints_PD[subject][action]['pos'][cam_idx] = kps
                keypoints_PD[subject][action]['conf'][cam_idx] = conf
            keypoints_PD[subject][action]['subject'] = np.ones((kps.shape[0])) * int(subject[1:])
    
    # Get subjs of interest
    ch0_data = np.concatenate([keypoints_PD[subj][task]['pos'][0] for subj in subjs])
    ch0_conf = np.concatenate([keypoints_PD[subj][task]['conf'][0] for subj in subjs])
    ch1_data = np.concatenate([keypoints_PD[subj][task]['pos'][1] for subj in subjs])
    ch1_conf = np.concatenate([keypoints_PD[subj][task]['conf'][1] for subj in subjs])
    
    if mohsens_output:
        # concat the subject data together for each channel
        out_poses_2d = np.array([ch0_data, ch1_data]) 
        out_confidences = np.array([ch0_conf, ch1_conf])
        out_subject = np.concatenate([keypoints_PD[subj][task]['subject'] for subj in subjs])
        return out_poses_2d, out_confidences, out_subject
    else:
        return ch0_data, ch1_data, ch0_conf, ch1_conf 
    

def filter_ap_detections(ap_preds, ch):
    '''
    My re-adaptation of Mohsens function to process alphapose pred json files

    args:
        data: the alphapose output json file loaded in as a dictionary (ex: data1 = json.load(f))
        ch: the channel of the camera (1, 2, 3, 4, 5, 6, 7, 8)
    '''
    filters = alphapose_filtering.AP_view_filters[ch]

    # regroup detections to be per-frame 
    frame_names = list(set([ap_preds[i]["image_id"] for i in range(len(ap_preds))]))
    def frame_name_sort_key(frame_name):
        return int(frame_name[:-4])
    frame_names.sort(key=frame_name_sort_key)
    frame_detections = {}
    for frame_name in frame_names:
        frame_detections[frame_name] = [ap_preds[i] for i in range(len(ap_preds)) if ap_preds[i]["image_id"] == frame_name]

    kpt = np.zeros((len(frame_names), 16, 3))
    conf_thresh = 0.6
    img_w, img_h = 3840, 2160
    x_edge_thresh = 50
    y_edge_thresh = 10

    # choose the detection to keep for each frame, if any
    for i, frame_name in enumerate(frame_names):
        if i == 900:
            foo = 0
        for detection in frame_detections[frame_name]:
            joint_idxs = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            # Filters
            bbx_x0, bbx_y0, bbx_dx, bbx_dy = detection['box']

            not_at_x_edge = (bbx_x0 > x_edge_thresh) and ((bbx_x0 + bbx_dx) < (img_w - x_edge_thresh))
            not_at_y_edge = (bbx_y0 > y_edge_thresh) and ((bbx_y0 + bbx_dy) < (img_h - y_edge_thresh))

            confident = np.mean(np.asarray(detection['keypoints'])[[idx*3 + 2 for idx in joint_idxs]]) > conf_thresh # pred confidence threshold
            
            # bbx width threshold
            if (filters['bbx_h_min'] is None):
                bbx_large = True
            else:
                bbx_large = (bbx_dy > filters['bbx_h_min'])     

            # subj in correct area in frame?
            if (filters['x_min'] is None):
                bbx_subj_loc = True
            else:
                bbx_subj_loc = (bbx_x0 > filters['x_min'])

            if (filters['x_max'] is None):
                bbx_subj_loc = bbx_subj_loc
            else:
                bbx_subj_loc = (bbx_x0 < filters['x_max']) and bbx_subj_loc

            # If detection is good, save it
            if (not_at_x_edge and not_at_y_edge and bbx_large and confident and bbx_subj_loc): 
                xy = np.zeros((16, 3)) 
                joints = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
                xy[joints, 0] = np.asarray(detection['keypoints'])[[idx*3 for idx in joint_idxs]]        # x
                xy[joints, 1] = np.asarray(detection['keypoints'])[[idx*3 + 1 for idx in joint_idxs]]    # y
                xy[joints, 2] = np.asarray(detection['keypoints'])[[idx*3 + 2 for idx in joint_idxs]]    # conf
                kpt[i] = xy
                # print("saved at frame {}".format(i+1))   
            else:     
                # print("  filtered at frame {}".format(i+1))
                pass
    return kpt

def filter_alphapose_results(data_path, subj, task, chs, kpts_dict=None, overwrite=False):
    '''
    Filters alphapose output jsons to keep the subject detections only.

    If new cam system, then the channels are processed together as a pair.

    If old cam system, the channels are processed separately and the poses dont necessarily correspond.

    TODO: MAKE IT WORK FOR ARBITRARY NUMBER OF CHANNEL INPUTS, NOT JUST PAIRS

    args:
        data_path: path to the alphapose json outputs (eg: dataset_path + "/9731/free_form_oval/")
        subj: subject ID for dict
        task: UPDRS task name
        channels: what channels to process (eg: [3, 4], [2, 7])     NOTE: USES NEW SYSTEM CHANNEL NAMING
        kpts_dict: dictionary of 2D keypoints data to be updated
        overwrite: whether to overwrite existing data in kpts_dict
    '''
    cam_ver = cam_sys_info.cam_ver[subj]
    print("cam_ver: {}".format(cam_ver))

    # Initialize the dictionary components as needed
    if kpts_dict is None:
        kpts_dict = {}
    if subj not in kpts_dict:
        kpts_dict[subj] = {}
    if task not in kpts_dict[subj]:
        kpts_dict[subj][task] = {'pos': {}, 'conf': {}}
    
    # Get kpts from channels
    kpts = []
    for ch in chs:
        # check if channel data already present, else initialize
        if (ch in kpts_dict[subj][task]['pos']) and (not overwrite):
            print("WARNING: Channel {} already present in kpts_dict, skipping".format(ch))
            return kpts_dict
        # else:
        #     kpts_dict[subj][task]['pos'][ch] = {}
        #     kpts_dict[subj][task]['conf'][ch] = {}
        
        # Load and process data from json file of camera channel
        ap_preds_json = '{}CH{}_alphapose-results.json'.format(data_path, ch if (cam_ver == 'new') else cam_sys_info.ch_new_to_old[ch])
        print("Loading {}".format(ap_preds_json))
        with open(ap_preds_json) as f:
            ap_results = json.load(f)
            kpts.append(filter_ap_detections(ap_results, ch))
    print("")

    kpts_1 = kpts[0]
    kpts_2 = kpts[1]

    # If processing together, trim to same length (may have an extra frame or two due to AP video processing)
    if cam_ver == 'new':
        kpts_1 = kpts_1[:min(kpts_1.shape[0], kpts_2.shape[0])]
        kpts_2 = kpts_2[:min(kpts_1.shape[0], kpts_2.shape[0])]

    print("kpt1.shape: {} kpt2.shape: {}".format(kpts_1.shape, kpts_2.shape))
    

    # Remove frames with 0 pose on channels
    if cam_ver == 'new':
        pose_exists = np.logical_and(np.sum(kpts_1, axis=(1,2)) != 0, np.sum(kpts_2, axis=(1,2)) != 0)
        pose_exists_1 = pose_exists
        pose_exists_2 = pose_exists
    elif cam_ver == 'old':
        pose_exists_1 = (np.sum(kpts_1, axis=(1,2)) != 0)
        pose_exists_2 = (np.sum(kpts_2, axis=(1,2)) != 0)
    
    kpts_1 = kpts_1[pose_exists_1]
    kpts_2 = kpts_2[pose_exists_2]

    print("EXISTANCE FILTERED: kpt1.shape: {} kpt2.shape: {}".format(kpts_1.shape, kpts_2.shape))
    
    if cam_ver == 'new': assert kpts_1.shape[0] == kpts_2.shape[0], "kpts_1.shape[0] != kpts_2.shape[0]"

    kpts_dict[subj][task]['pos'][0] = kpts_1[:, :, :2]
    kpts_dict[subj][task]['pos'][1] = kpts_2[:, :, :2]
    kpts_dict[subj][task]['conf'][0] = kpts_1[:, :, 2:]
    kpts_dict[subj][task]['conf'][1] = kpts_2[:, :, 2:]

    return kpts_dict    


class body_ts_loader():
    '''
    For loading the extracted 3D body keypoints from the UPDRS dataset
    '''
    def __init__(self, ts_path, subjects = info.subjects_All) -> None:
        self.subjects = subjects
        self.ts_path = ts_path
        self.feat_names = info.clinical_gait_feat_names

        self.data_normal = []
        for idx, subj in enumerate(subjects):
            data_normal = np.load(ts_path + 'Predictions_' + subj + '.npy')     # (num_frames, num_joints, 3)
            for ii in range(15):
                for jj in range(3):
                    data_normal[:,ii,jj] = savgol_filter(data_normal[:,ii,jj], 11, 3)   # Smoothing

            x_vec = data_normal[:,1] - data_normal[:,4]      # R Hip - L Hip
            y_vec = data_normal[:,7] - data_normal[:,0]      # Neck - Pelvis
            x_vec /= np.linalg.norm(x_vec, keepdims=True, axis=-1)
            y_vec /= np.linalg.norm(y_vec, keepdims=True, axis=-1)
            z_vec = np.cross(x_vec, y_vec)

            rotation_matrix = np.ones((len(x_vec), 3, 3))
            rotation_matrix[:,:,0] = x_vec
            rotation_matrix[:,:,1] = y_vec
            rotation_matrix[:,:,2] = z_vec

            self.data_normal.append(np.matmul(data_normal, rotation_matrix))

    # Get the norm timeseries data for a specific subject
    def get_data_norm(self, subject):
        return self.data_normal[self.subjects.index(subject)]
   
class PD_AP_Dataset(Dataset):
    '''
    Dataset for the 2D Alphapose keypoints extracted from the UPDRS videos
    '''

    def __init__(self, poses_2d, confidences, subject):
        self.poses_2d = poses_2d
        self.conf = confidences
        self.subjects = subject

    def __len__(self):
        return self.poses_2d[0].shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = dict()
        sample['confidences'] = dict()

        for c_idx in range(len(self.poses_2d)):
            p2d = torch.Tensor(self.poses_2d[c_idx][idx].astype('float32')).cuda()
            sample['confidences'][c_idx] = torch.Tensor(self.conf[c_idx][idx].astype('float32')).squeeze().cuda()
            sample['cam' + str(c_idx)] = p2d
            
        sample['subjects'] = self.subjects[idx]

        return sample
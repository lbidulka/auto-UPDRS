import torch
from torch.utils.data import Dataset
from scipy.signal import savgol_filter
import numpy as np
import json
import os
import pickle

from utils import info, alphapose_filtering, cam_sys_info

def get_2D_data(subjs, tasks, data_path, normalized=True, mohsens_data=False, mohsens_output=False,
                old_sys_return_only=0):
    '''
    Fetches the 2D pose data for specified subjects.

    TODO: ADD SUPPORT FOR ARBITRARY CHANNELS

    Args:
        subj (str):                         subject ID
        tasks (list of str):                list of task names (e.g. ['free_form_oval', ...])
        chs (list):                         list of channels to fill (e.g. ['001', '002',])    NOTE: THIS USES NEW SYS NAMING
        data_path (str):                    path to the 2d kpts dataset
        normalize (bool):                   if True, normalize the data
        mohsens_data (bool):                if True, load differently for Mohsens data, else use as is
        mohsens_output (bool):              if True, return the data in the same format as mohsens data
        old_sys_return_only (int):          for old system subjs, return data for this channel (0 or 1) for both outputs
    Returns: 
        ch0_data (np.array):                2D pose data for view 0
        ch1_data (np.array):                2D pose data for view 1
    '''
    # Load up all subjs
    keypoints_PD = np.load(data_path, allow_pickle=True)
    if mohsens_data:
        tasks = ['WalkingOval']
        keypoints_PD = keypoints_PD['positions_2d'].item()  # contains all subjects

    for subject in keypoints_PD.keys():
        for action in keypoints_PD[subject]:
            keypoints_PD[subject][action]['subject'] = {}
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
                keypoints_PD[subject][action]['subject'][cam_idx] = np.ones((kps.shape[0])) * int(subject[1:])
    
    # Get data for subjs of interest

    ch_data = []
    ch_conf = []
    ch_subject = []
    ch_frames = []
    for cam_idx in range(len(keypoints_PD[subjs[0]][tasks[0]]['pos'])):
        ch_data.append(np.concatenate([keypoints_PD[subj][task]['pos'][cam_idx] for subj in subjs for task in tasks]))
        ch_conf.append(np.concatenate([keypoints_PD[subj][task]['conf'][cam_idx] for subj in subjs for task in tasks]))
        ch_subject.append(np.concatenate([keypoints_PD[subj][task]['subject'][cam_idx] for subj in subjs for task in tasks]))
        ch_frames.append(np.concatenate([keypoints_PD[subj][task]['idx'][cam_idx] for subj in subjs for task in tasks]))


    # ch0_data = np.concatenate([keypoints_PD[subj][task]['pos'][0] for subj in subjs for task in tasks])
    # ch0_conf = np.concatenate([keypoints_PD[subj][task]['conf'][0] for subj in subjs for task in tasks])
    # ch0_subject = np.concatenate([keypoints_PD[subj][task]['subject'][0] for subj in subjs for task in tasks])
    # ch0_frames = np.concatenate([keypoints_PD[subj][task]['idx'][0] for subj in subjs for task in tasks])

    # ch1_data = np.concatenate([keypoints_PD[subj][task]['pos'][1] for subj in subjs for task in tasks])
    # ch1_conf = np.concatenate([keypoints_PD[subj][task]['conf'][1] for subj in subjs for task in tasks])
    # ch1_subject = np.concatenate([keypoints_PD[subj][task]['subject'][1] for subj in subjs for task in tasks])
    # ch1_frames = np.concatenate([keypoints_PD[subj][task]['idx'][1] for subj in subjs for task in tasks])

    # For the old system, we just return the data for one channel (as per Mohsens method)
    # TODO: DO WE NEED TO DO THIS HERE? OR CAN WE JUST DO IT OUTSIDE, TO BE LESS CONFUSING?
    # TODO: UPDATE FOR ARBITRARY NUM CHANNELS
    if cam_sys_info.cam_ver[subjs[0]] == 'old':
        if old_sys_return_only == 0:
            ch1_data = ch0_data
            ch1_conf = ch0_conf
            ch1_subject = ch0_subject
            ch1_frames = ch0_frames
        elif old_sys_return_only == 1:
            ch0_data = ch1_data
            ch0_conf = ch1_conf
            ch0_subject = ch1_subject
            ch0_frames = ch1_frames
        else:
            raise ValueError("old_sys_return_ch must be 0 or 1")
    
    # if mohsens_output:
    # concat the subject data together for each channel
    out_poses_2d = np.array(ch_data)
    out_confidences = np.array(ch_conf)
    out_subject = np.array(ch_subject[0])
    out_frames = np.array(ch_frames)

        
        # out_poses_2d = np.array([ch0_data, ch1_data]) 
        # out_confidences = np.array([ch0_conf, ch1_conf])
        # out_subject = np.array(ch0_subject) if old_sys_return_only == 0 else np.array(ch1_subject)
        # return out_poses_2d, out_confidences, out_subject
    # else:
    #     return ch0_data, ch1_data, ch0_conf, ch1_conf, ch0_frames, ch1_frames

    return (out_poses_2d, out_confidences, out_subject) if mohsens_output else (out_poses_2d, out_confidences, out_subject, out_frames)
    

def filter_ap_detections(ap_preds, ch, subj, keep_halpe=False):
    '''
    My re-adaptation of Mohsens function to process alphapose pred json files

    args:
        data: the alphapose output json file loaded in as a dictionary (ex: data1 = json.load(f))
        ch: the channel of the camera (1, 2, 3, 4, 5, 6, 7, 8)
        subj: subject ID
    '''
    bbx_filters = alphapose_filtering.AP_bbx_filters[ch]

    # Exceptions
    if cam_sys_info.cam_ver[subj] == 'old':
        if ch == '006':
            bbx_filters['x_max'] = None # Evaluator is not in frame for these (though I havnt check every single one...)

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
    y_edge_thresh = 75 # TODO: CHOOSE GOOD VAL HERE

    # choose the detection to keep for each frame, if any
    for i, frame_name in enumerate(frame_names):
        for detection in frame_detections[frame_name]:
            joint_idxs = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
            #19*3,11*3,13*3,15*3,12*3,14*3,16*3,18*3,17*3,6*3,8*3,10*3,5*3,7*3,9*3
            joint_idxs = [19, 11, 13, 15, 12, 14, 16, 18, 17, 6, 8, 10, 5, 7, 9]
            # Filters
            bbx_x0, bbx_y0, bbx_dx, bbx_dy = detection['box']

            not_at_x_edge = (bbx_x0 > x_edge_thresh) and ((bbx_x0 + bbx_dx) < (img_w - x_edge_thresh))
            not_at_y_edge = (bbx_y0 > y_edge_thresh) and ((bbx_y0 + bbx_dy) < (img_h - y_edge_thresh))

            confident = np.mean(np.asarray(detection['keypoints'])[[idx*3 + 2 for idx in joint_idxs]]) > conf_thresh # pred confidence threshold
            
            # bbx height threshold
            if (bbx_filters['bbx_h_min'] is None):
                bbx_large = True
            else:
                bbx_large = (bbx_dy > bbx_filters['bbx_h_min'])     

            # subj in correct area in frame?
            if (bbx_filters['x_min'] is None):
                bbx_subj_loc = True
            else:
                bbx_subj_loc = (bbx_x0 > bbx_filters['x_min'])

            if (bbx_filters['x_max'] is None):
                bbx_subj_loc = bbx_subj_loc
            else:
                bbx_subj_loc = (bbx_x0 < bbx_filters['x_max']) and bbx_subj_loc

            # If detection is good, save it
            if (not_at_x_edge and not_at_y_edge and bbx_large and confident and bbx_subj_loc): 
                xy = np.asarray(detection['keypoints']).reshape(-1, 3)
                if not keep_halpe:
                    xy = np.insert(xy[joint_idxs], 7, [0,0,0], axis=0)
                kpt[i] = xy
                # print("saved at frame {}".format(i+1))   
            else:     
                # print("  filtered at frame {}".format(i+1))
                pass
    return kpt

def filter_alphapose_results(data_path, subj, task, chs, kpts_dict=None, overwrite=False, keep_halpe=False):
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
        kpts_dict[subj][task] = {'pos': {}, 'conf': {}, 'idx': {}}
    
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
        # print("Loading {}".format(ap_preds_json))
        with open(ap_preds_json) as f:
            ap_results = json.load(f)
            kpts.append(filter_ap_detections(ap_results, ch, subj, keep_halpe=keep_halpe))
    print("")

    kpts_idxs = []
    pose_exists = []

    kpts_lens = [pts.shape[0] for pts in kpts]
    for i, ch in enumerate(chs):
        # If processing together, trim to same length (may have an extra frame or two due to AP video processing)
        if cam_ver == 'new':
            kpts[i] = kpts[i][:min(kpts_lens)]

        print("kpts[{}].shape: {}".format(i, kpts[i].shape))

        # Keep only frames from the task performance (only works for new cam system)
        if cam_ver == 'new':
            fps = 30
            start_frame = alphapose_filtering.task_timestamps[subj][task]['start'] * fps
            end_frame = alphapose_filtering.task_timestamps[subj][task]['end'] * fps

            if subj == 'S31' and task == 'free_form_oval' and ch == '002':
                start_frame -= 25
                end_frame -= 25

            kpts[i] = kpts[i][start_frame:end_frame]

            print("TASK TIME FILTERED kpts[{}].shape: {}".format(i, kpts[i].shape))

        # Get frame idx for each pred, for timestamping
        kpts_idxs.append(np.arange(kpts[i].shape[0]))

        # Find valid poses
        pose_exists.append(np.sum(kpts[i], axis=(1,2)) != 0)
        
    # Find frames with no invalid poses for any channel, if its training data
    if cam_ver == 'new':
        paired_frames = np.logical_and(pose_exists[0], pose_exists[1])
        if len(pose_exists) > 2:
            for i in range(2, len(pose_exists)):
                paired_frames = np.logical_and(paired_frames, pose_exists[i])
    
    # Trim to valid data
    for i, ch, in enumerate(chs):
        kpts[i] = kpts[i][paired_frames if cam_ver == 'new' else pose_exists[i]]
        kpts_idxs[i] = kpts_idxs[i][paired_frames if cam_ver == 'new' else pose_exists[i]]

        # TEMP: if no poses found, insert a single all-0 pose (specifically, TUG task S21 vid is too short)
        if (kpts[i].shape[0] == 0):
            print("\n!!! ---> WARNING: No poses found for subj: {}, cam: {}, task: {}".format(subj, i, task))
            print("inserting a single all-0 pose...\n")
            kpts[i] = np.zeros((1, 16, 3)) + 1e-6
            kpts_idxs[i] = np.array([0])
        
        print("EXISTANCE FILTERED kpts[{}].shape: {}".format(i, kpts[i].shape))

        # Save for output
        kpts_dict[subj][task]['pos'][i] = kpts[i][:, :, :2]    
        kpts_dict[subj][task]['conf'][i] = kpts[i][:, :, 2:]
        kpts_dict[subj][task]['idx'][i] = kpts_idxs[i]

    return kpts_dict    


class body_ts_loader():
    '''
    For loading the extracted 3D body keypoints from the UPDRS dataset
    '''
    def __init__(self, ts_path, in_2d_path, task, subjects=info.subjects_All, pickled=False, proc_aligned=True,
                smoothing=True, zero_rot=True,  # normalization options
                ) -> None:
        '''
            args:
            pickled: if False, loads the individual numpy files (Mohsens style), else loads the pickled dictionary (my style)
            task: the UPDRS task to load    # TODO: ADD SUPPORT FOR MULTIPLE TASKS
        '''
        self.task = task
        self.subjects = subjects
        self.in_2d_path = in_2d_path
        self.ts_path = ts_path
        self.feat_names = info.clinical_gait_feat_names

        if pickled:
            with open(ts_path, 'rb') as f:
                all_body_3d_preds = pickle.load(f)

        self.data_normal = []
        self.data_idxs = []
        for idx, subj in enumerate(subjects):
            # Load as needed
            if pickled:
                data_normal = all_body_3d_preds[subj][task]['aligned_3d_preds'] if proc_aligned else all_body_3d_preds[subj][task]['raw_3d_preds']
                subj_frame_idxs = all_body_3d_preds[subj][task]['frame_idxs']
            else:
                data_normal = np.load(ts_path + 'Predictions_' + subj + '.npy')     # (num_frames, num_joints, 3)

            if smoothing:
                for ii in range(15):
                    for jj in range(3):
                        data_normal[:,ii,jj] = savgol_filter(data_normal[:,ii,jj], 11, 3)   # Smoothing

            # Make poses relative to the first pose
            if zero_rot: # and not proc_aligned:
                Rhip_idx = info.PD_3D_skeleton_kpt_idxs['RHip']
                Lhip_idx = info.PD_3D_skeleton_kpt_idxs['LHip']
                Neck_idx = info.PD_3D_skeleton_kpt_idxs['Neck']
                Hip_idx = info.PD_3D_skeleton_kpt_idxs['Hip']

                x_vec = data_normal[:, Lhip_idx] - data_normal[:, Rhip_idx]      # L Hip - R Hip
                y_vec = data_normal[:, Neck_idx] - data_normal[:, Hip_idx]      # Neck - Pelvis
                x_vec /= np.linalg.norm(x_vec, keepdims=True, axis=-1)
                y_vec /= np.linalg.norm(y_vec, keepdims=True, axis=-1)
                z_vec = np.cross(x_vec, y_vec)

                rotation_matrix = np.ones((len(x_vec), 3, 3))
                # Only use first pose
                rotation_matrix[:,:,0] = x_vec #np.repeat(x_vec[0].reshape((-1,3)), len(x_vec), axis=0) # x_vec
                rotation_matrix[:,:,1] = y_vec #np.repeat(y_vec[0].reshape((-1,3)), len(y_vec), axis=0) # y_vec
                rotation_matrix[:,:,2] = z_vec #np.repeat(z_vec[0].reshape((-1,3)), len(z_vec), axis=0) # z_vec

                # Rot the pose back to centre
                data_normal = np.matmul(data_normal, rotation_matrix)

            self.data_normal.append(data_normal)
            self.data_idxs.append(subj_frame_idxs)

    # Get the norm timeseries data for a specific subject
    def get_data_norm(self, subject):
        return self.data_normal[self.subjects.index(subject)]
   
class PD_AP_Dataset(Dataset):
    '''
    Dataset for the 2D Alphapose keypoints extracted from the UPDRS videos
    '''

    def __init__(self, poses_2d, confidences, subject, frame_idxs=None):
        self.poses_2d = poses_2d
        self.conf = confidences
        self.subjects = subject
        if frame_idxs is not None:
            self.frame_idxs = frame_idxs

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
        if hasattr(self, 'frame_idxs'):
            sample['frame_idxs'] = {}
            for c_idx in range(len(self.poses_2d)):
                sample['frame_idxs'][c_idx] = self.frame_idxs[c_idx][idx]

        return sample
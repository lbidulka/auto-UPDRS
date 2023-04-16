from utils import info, alphapose_filtering
from scipy.signal import savgol_filter
import numpy as np
import json
import os

def get_2D_data(subj, task, data_path='auto_UPDRS/data/body/2d_proposals/mohsen_data_PD.npz', 
                mohsens_data=False, mohsens_output=False):
    '''
    Fetches the 2D pose data for a given subject
    TODO: CHANGE TO A CLASS AND DONT DO THE LOADING EVERY TIME WE CALL THIS
    TODO: IMPLEMENT MULTIPLE SUBJECTS
    Args:
        subj (str): subject ID
        task (str): task name (e.g. 'free_form_oval', ...)
        data_path (str): path to the 2d kpts dataset
        mohsens_data (bool): if True, load differently for Mohsens data, else use as is
        mohsens_output (bool): if True, return the data in the same format as mohsens data
    Returns: 
        ch3_data (np.array): 2D pose data for channel 3
        ch4_data (np.array): 2D pose data for channel 4
        TODO: ADD CONFIDENCE DATA
    '''
    # Load up all subjs
    keypoints_PD = np.load(data_path, allow_pickle=True)
    if mohsens_data:
        task = 'WalkingOval'
        keypoints_PD = keypoints_PD['positions_2d'].item()  # contains all subjects

    for subject in keypoints_PD.keys():
        for action in keypoints_PD[subject]:
            for cam_idx in range(len(keypoints_PD[subject][action]['pos'])):
                kps = keypoints_PD[subject][action]['pos'][cam_idx]     # (n_frames, n_kpts, xy)
                conf = keypoints_PD[subject][action]['conf'][cam_idx]

                kps = kps[:,[0,1,2,3,4,5,6,8,9,10,11,12,13,14,15],:]
                conf = conf[:,[0,1,2,3,4,5,6,8,9,10,11,12,13,14,15],:]

                kps = np.transpose(kps,[0,2,1])
                kps = kps.reshape(-1, 15*2)  # flat: (1, xxx ... yyy)
                
                keypoints_PD[subject][action]['pos'][cam_idx] = kps
                keypoints_PD[subject][action]['conf'][cam_idx] = conf
    
    # Get subjs of interest
    ch3_data = keypoints_PD[subj][task]['pos'][0] 
    ch3_conf = keypoints_PD[subj][task]['conf'][0] 
    ch4_data = keypoints_PD[subj][task]['pos'][1]  
    ch4_conf = keypoints_PD[subj][task]['conf'][1]
    
    # TODO: VERIFY MOHSENS OUTPUT WORKS WITH HIS TRAINING CODE
    if mohsens_output:
        # concat the subject data together for each channel
        out_poses_2d = [np.concatenate(data, axis=0) for data in zip(ch3_data, ch4_data)]
        out_confidences = [np.concatenate(data, axis=0) for data in zip(ch3_conf, ch4_conf)]
        return out_poses_2d, out_confidences
    else:
        return ch3_data, ch4_data, ch3_conf, ch4_conf 
    

def get_AP_pred_filter_settings(channel, camera_ver):
    '''
    Selects the appropriate settings for filtering the AP predictions, based on what channel 
    and camera version we are using

    NOTE: "new" system channels correspond to diff views in the room than "old" system channels

    TODO: MAKE THIS INTO A DICT INSTEAD

    returns:
        x_min (int):        lower bbox x coord threshold, if any
        x_max (int):        upper bbox x coord threshold, if any
        bbx_w_min (int):  lower bbox width threshold, if any
    '''
    x_min, x_max, bbx_w_min, bbx_w_max = None, None, None, None
    
    if channel == '002':
        if (camera_ver == 'new'):
            x_max = 1700
        else:
            raise NotImplementedError
        
    elif channel == '003':
        if (camera_ver == 'new'):
            x_min = 2800
        elif (camera_ver == 'old'):
            x_min = 2700
        bbx_w_min = 1000 # Threshold for the width of the bounding box, evaluator has ~630

    elif channel == '004':
        if (camera_ver == 'new'):
            x_min = 2800
        elif (camera_ver == 'old'):
            x_min = 2700
        bbx_w_min = 1000 # Threshold for the width of the bounding box, evaluator has ~630

    elif channel == '006':
        if (camera_ver == 'new'):
            x_max = 2920
        else:
            raise NotImplementedError

    elif channel == '007':
        if (camera_ver == 'new'):
            x_min = 960
        else:
            raise NotImplementedError
        
    return (x_min, x_max, bbx_w_min, bbx_w_max)

def filter_ap_detections(ap_preds, channel, camera):
    '''
    My re-adaptation of Mohsens function to process alphapose pred json files

    args:
        data: the alphapose output json file loaded in as a dictionary (ex: data1 = json.load(f))
        channel: the channel of the camera (1, 2, 3, 4, 5, 6, 7, 8)
        camera: the camera (old or new)
    '''
    (x_min, x_max, bbx_w_min, bbx_w_max) = get_AP_pred_filter_settings(channel, camera)

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
            
            bbx_large = True if (bbx_w_min is None) else (bbx_dx > bbx_w_min)     # bbx width threshold

            # subj in correct area in frame?
            bbx_subj_loc = True if (x_min is None) else (bbx_x0 > x_min)
            bbx_subj_loc = bbx_subj_loc if (x_max is None) else ((bbx_x0 < x_max) and bbx_subj_loc)

            # If detection is good, save it
            if not_at_x_edge and not_at_y_edge and bbx_large and confident and bbx_subj_loc: 
                xy = np.zeros((16, 3)) 
                joints = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
                xy[joints, 0] = np.asarray(detection['keypoints'])[[idx*3 for idx in joint_idxs]]        # x
                xy[joints, 1] = np.asarray(detection['keypoints'])[[idx*3 + 1 for idx in joint_idxs]]    # y
                xy[joints, 2] = np.asarray(detection['keypoints'])[[idx*3 + 2 for idx in joint_idxs]]    # conf
                kpt[i] = xy
                # print("saved at frame {}".format(i))       
            elif np.sum(kpt[i]) == 0:
                # print("filtered at frame {}".format(i))
                kpt[i] = np.zeros((16, 3))
    return kpt

def filter_alphapose_results(data_path, subj, task, channels, kpts_dict=None):
    '''
    Filters alphapose output jsons to keep the subject detections only

    args:
        data_path: path to the alphapose json outputs (eg: dataset_path + "/9731/free_form_oval/")
        subj: subject ID for dict
        task: UPDRS task name
        channels: what channels to process (eg: [3, 4], [2, 7])
        kpts_dict: dictionary of 2D keypoints data to be updated
    '''
    # Initialize the dictionary components as needed
    if kpts_dict is None:
        kpts_dict = {}
    if subj not in kpts_dict:
        kpts_dict[subj] = {}
    if task not in kpts_dict[subj]:
        kpts_dict[subj][task] = {'pos': {}, 'conf': {}}

    # Load and process data from json file of camera channels
    ch = channels[0] #'003'
    with open(os.path.join(data_path, 'CH' + ch + '_alphapose-results.json')) as f:
        cam1_ap_results = json.load(f)
        kpts_1 = filter_ap_detections(cam1_ap_results, ch, alphapose_filtering.cam_ver[subj])

    ch = channels[1] #'004'
    with open(os.path.join(data_path, 'CH' + ch + '_alphapose-results.json')) as f:
        cam2_ap_results = json.load(f)
        kpts_2 = filter_ap_detections(cam2_ap_results, ch, alphapose_filtering.cam_ver[subj])

    # Make sure same length, may have an extra frame or two due to original AP video processing
    kpts_1 = kpts_1[:min(kpts_1.shape[0], kpts_2.shape[0])]
    kpts_2 = kpts_2[:min(kpts_1.shape[0], kpts_2.shape[0])]

    print("kpt1.shape: ", kpts_1.shape, " kpt2.shape: ", kpts_2.shape)
    
    # TODO: JUST DUPLICATE CH003 INTO CH004 FOR NON-TRAINING DATA LIKE MOHSEN

    # Remove frames with 0 pose on any channel
    pose_exists = np.logical_and(np.sum(kpts_1, axis=(1,2)) != 0, np.sum(kpts_2, axis=(1,2)) != 0)
    kpts_1 = kpts_1[pose_exists]
    kpts_2 = kpts_2[pose_exists]

    print("EXISTANCE FILTERED  kpt1.shape: ", kpts_1.shape, " kpt2.shape: ", kpts_2.shape)
    
    assert kpts_1.shape[0] == kpts_2.shape[0], "kpts_1.shape[0] != kpts_2.shape[0]"

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
   

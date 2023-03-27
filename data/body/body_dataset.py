from utils import info, alphapose_filtering
from scipy.signal import savgol_filter
import numpy as np
import json
import os

def get_2D_data(subj, task, data_path='auto_UPDRS/data/body/2d_proposals/mohsen_data_PD.npz', mohsens_data=False):
    '''
    Fetches the 2D pose data for a given subject

    Args:
        subj (str): subject ID
        task (str): task name (e.g. 'free_form_oval', ...)
        data_path (str): path to the 2d kpts dataset
        mohsens_data (bool): if True, load differently for Mohsens data, else use as is
    Returns: 
        ch3_data (np.array): 2D pose data for channel 3
        ch4_data (np.array): 2D pose data for channel 4
        TODO: ADD CONFIDENCE DATA
    '''
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
    
    ch3_data = keypoints_PD[subj][task]['pos'][0]  
    ch4_data = keypoints_PD[subj][task]['pos'][1]  
    
    return ch3_data, ch4_data

def MOHSEN_process_data(data, channel, camera, len_data=4000):
    '''
    My adaptation of Mohsens function to process alphapose pred json files

    args:
        data: the alphapose output json file loaded in as a dictionary (ex: data1 = json.load(f))
        channel: the channel of the camera (1 or 2)
        camera: the camera (old or new)
        len_data: not sure, but always is set to 4000 (except for S01, where its 4046) TODO: FIND OUT WHAT THIS IS
    '''
    # X pos filter, to remove the evaluator
    if channel == 1 and camera == 'new':
        crop = 2000 #2800
    elif channel == 1 and camera == 'old':
        crop = 2000 #2700  
    elif channel == 2 and camera == 'new':
        crop = 2000
    elif channel == 2 and camera == 'old':
        crop = 1950    

    bbx_ = np.zeros(len_data)
    kpt = np.zeros((len_data, 16, 3))

    # TODO: UNDERSTAND THIS WEIRD LOOPING/PADDING
    for jj in range(0, len(data)):
        for frame_idx in range(0, len_data):
            if int(data[jj]['image_id'][:-4]) == (frame_idx + 1):
                # joint_idxs = [19, 11, 13, 15, 12, 14, 16, 18, 17, 6, 8, 10, 5, 7, 9] # TODO: WHAT KPTS ARE THESE?
                joint_idxs = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

                conf = np.mean(np.asarray(data[jj]['keypoints'])[[idx*3 + 2 for idx in joint_idxs]])
                bbx = abs(data[jj]['box'][2] - data[jj]['box'][0])

                # Filters
                in_subj_region = data[jj]['keypoints'][5*3] < crop  # x-axis crop out the evaluator in the frame (L-Shoulder? kpt)
                confident = conf > 0.6                              # pred confidence threshold
                # TODO: UNDERSTAND THIS ONE
                # I think it checks last valid detection and makes sure we are moving
                # in same direction (increasing x-coord)
                bbx_reg = bbx_[frame_idx] < bbx                     

                if in_subj_region and bbx_reg and confident: 
                    bbx_[frame_idx] = bbx
                    xy = np.zeros((16, 3)) 
                    joints = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
                    
                    xy[joints, 0] = np.asarray(data[jj]['keypoints'])[[idx*3 for idx in joint_idxs]]
                    xy[joints, 1] = np.asarray(data[jj]['keypoints'])[[idx*3 + 1 for idx in joint_idxs]]
                    xy[joints, 2] = np.asarray(data[jj]['keypoints'])[[idx*3 + 2 for idx in joint_idxs]]

                    # xy[7, :] = (xy[0, :] + xy[8, :]) / 2    # TODO: WHAT IS THIS? IS IT FOR ADAPTING TO DIFF KPT FORMAT?
                    kpt[frame_idx] = xy          
    return kpt

def MOHSEN_filter_alphapose_results(data_path, subj, task, kpts_dict=None):
    '''
    Fixed-up implementation of Mohsens alphapose output filtering

    args:
        data_path: path to the alphapose json output (eg: dataset_path + "/9731/free_form_oval/CH003_alphapose-results.json")
        subj: subject ID for dict
        task: UPDRS task name
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
    ch = '003'
    with open(os.path.join(data_path, 'CH' + ch + '_alphapose-results.json')) as f:
        cam1_ap_results = json.load(f)
        kpts_1 = MOHSEN_process_data(cam1_ap_results, 1, alphapose_filtering.cam_ver[subj])

    ch = '004'
    with open(os.path.join(data_path, 'CH' + ch + '_alphapose-results.json')) as f:
        cam2_ap_results = json.load(f)
        kpts_2 = MOHSEN_process_data(cam2_ap_results, 2, alphapose_filtering.cam_ver[subj])

    # Filter out bad frames
    # TODO: JUST DUPLICATE CH003 INTO CH004 FOR NON-TRAINING DATA LIKE MOHSEN
    print("kpt1.shape: ", kpts_1.shape, " kpt2.shape: ", kpts_2.shape)

    row1 = alphapose_filtering.ts_row1_filters[subj]
    row2 = alphapose_filtering.ts_row2_filters[subj]

    kpts_1 = kpts_1[row1]
    kpts_2 = kpts_2[row2]

    print("ROW FILTERED  kpt1.shape: ", kpts_1.shape, " kpt2.shape: ", kpts_2.shape)

    # TODO: UNDERSTAND THIS WEIRD FILTERING
    # TODO: MAKE IT WORK FOR NOT JUST TRAINING SUBJS
    if subj in ['S01']:
        pass
    elif subj in ['S02', ]:
        row = (kpts_1[:,0,1] > 10) * (kpts_2[:,0,1] > 10)
    elif subj in ['S25', 'S26',]:
        row = (kpts_1[:,0,0] > 10)
    elif subj in ['S27', 'S28', 'S29',]:
        row = (kpts_1[:,0,1] > 100) * (kpts_2[:,0,1] > 100)

    kpts_1, kpts_2 = kpts_1[row], kpts_2[row]

    print("SECOND FILTERED  kpt1.shape: ", kpts_1.shape, " kpt2.shape: ", kpts_2.shape)

    # Save the kpts and confidences to the 2 channels
    kpts_dict[subj][task]['pos'][0] = kpts_1[:, :, :2]
    kpts_dict[subj][task]['pos'][1] = kpts_2[:, :, :2]
    kpts_dict[subj][task]['conf'][0] = kpts_1[:, :, 2:]
    kpts_dict[subj][task]['conf'][1] = kpts_2[:, :, 2:]

    return kpts_dict    


def filter_alphapose_preds(alphapose_results, cam_idx):
    '''
    I hate this function, but it works for now. It tries to remove the bad alphapose predictions + predictions for the wrong person
    '''
    # regroup detections to be per-frame 
    frame_names = set([alphapose_results[i]["image_id"] for i in range(len(alphapose_results))])
    frame_detections = {}
    for frame_name in frame_names:
        frame_detections[frame_name] = [alphapose_results[i] for i in range(len(alphapose_results)) if alphapose_results[i]["image_id"] == frame_name]
    
    # reject proposal if pelvis x coord is on right hand side, as its probably the wrong person. That is,
    # if bounding box centre (x,y) has: (x > 3/4 of image width) and (y < 1/3 of image height)
    # 
    # also reject if score is too low
    img_w = 3840     # TODO: WHY DO THE DETECTION COORDS NOT SEEM TO LINE UP WITH THE IMG DIMS? IS ALPHAPOSE RESIZING?
    img_h = 2160 
    if cam_idx == '003':
        x_rej_thresh = 1/2 * img_w
        y_rej_thresh = 3/4 * img_h
    elif cam_idx == '004':
        x_rej_thresh = 2/3 * img_w
        y_rej_thresh = 2/3 * img_h

    score_rej_thresh = 2.25  # TODO: DETERMINE A GOOD THRESHOLD TO USE?
    bbox_centre_dist_thresh = 150  # TODO: DETERMINE A GOOD THRESHOLD TO USE?

    # choose the detection to keep for each frame, if any
    for i, frame_name in enumerate(frame_names):
        # get the bounding boxes for the detections in this frame
        # bboxes have format: [[x_min, y_min], [width, height]] # TODO: VERIFY THIS
        bnd_boxes = np.array([np.array(frame_detections[frame_name][j]["box"]).reshape(-1,2) for j in range(len(frame_detections[frame_name]))])
        # get the centre(s) of the bounding box(es)
        # bbox_x = bnd_boxes[:, 0, 0] + (bnd_boxes[:, 1, 0] - bnd_boxes[:, 0, 0]) / 2
        # bbox_y = bnd_boxes[:, 0, 1] + (bnd_boxes[:, 1, 1] - bnd_boxes[:, 0, 1]) / 2

        # Get the x_min and y_min of the bounding boxes
        bbox_x = bnd_boxes[:, 0, 0]
        bbox_y = bnd_boxes[:, 0, 1]
        bbox_h = bnd_boxes[:, 1, 1]

        # print("-------")
        # print([np.array(frame_detections[frame_name][j]["box"]) for j in range(len(frame_detections[frame_name]))])
        # print(bnd_boxes)
        # print(bnd_boxes[:, 0, 0], bnd_boxes[:, 1, 0]) 
        # print(bnd_boxes[:, 0, 1], bnd_boxes[:, 1, 1])

        # print("\nrej thresh: ", x_rej_thresh, y_rej_thresh,"     bbox centres (x,y): ", bbox_x, bbox_y)
        # mags = np.sqrt(bbox_x**2 + bbox_y**2)
        # print(" bbx mags: ", mags)
        
        # create frame position rejection mask
        # 
        # rej if in wrong area of image,
        x_rej = (bbox_x > x_rej_thresh)
        y_rej = (bbox_y < y_rej_thresh)
        # rej if (xmin < 10) or (ymin + height > 2150)
        x_edge_rej = (bbox_x < 10)
        y_edge_rej = (bbox_y + bbox_h > 2150)
        # print(" x_rej: ", x_rej, "  y_rej: ", y_rej)
        # print(" x_edge_rej: ", x_edge_rej, "  y_edge_rej: ", y_edge_rej)
        # rej_mask = ((x_rej & y_rej) | x_edge_rej | y_edge_rej)
        rej_mask = (x_rej | x_edge_rej | y_edge_rej)
        # print(" xy rej_mask: ", rej_mask)
        rej_mask = np.where(rej_mask)[0].tolist()
        # print(" xy rej_mask idxs: ", rej_mask)

        # reject detections
        # print(len(frame_detections[frame_name]))
        frame_detections[frame_name] = [frame_detections[frame_name][j] for j in range(len(frame_detections[frame_name])) 
                                        if j not in rej_mask]
        # also reject if score is too low
        # print(len(frame_detections[frame_name]))
        frame_detections[frame_name] = [frame_detections[frame_name][j] for j in range(len(frame_detections[frame_name])) 
                                        if frame_detections[frame_name][j]["score"] > score_rej_thresh]
        # print(len(frame_detections[frame_name]))
        
        # # reject if head is not above hip
        # if frame_detections[frame_name] != []:
        #     head_coords = [np.array(frame_detections[frame_name][j]["keypoints"]).reshape(-1,3)[5:20, :2][12]
        #                    for j in range(len(frame_detections[frame_name]))]
        #     hip_coords = [np.array(frame_detections[frame_name][j]["keypoints"]).reshape(-1,3)[5:20, :2][14]
        #                    for j in range(len(frame_detections[frame_name]))]

        #     print(" head: ", head_coords, "\n hip: ", hip_coords)

        # For now, just pick on if a few are close together
        # TODO: FIGURE OUT THESE DETECTIONS THAT ARE WITHIN A FEW PIXELS OF EACH OTHER
        if len(frame_detections[frame_name]) > 1:
            # get the bounding boxes for the remaining detections
            bnd_boxes = np.array([np.array(frame_detections[frame_name][j]["box"]).reshape(-1,2) for j in range(len(frame_detections[frame_name]))])
            # idxs = np.array([np.array(frame_detections[frame_name][j]["idx"]) for j in range(len(frame_detections[frame_name]))])
            # get the centre(s) of the bounding box(es)
            bbox_x = bnd_boxes[:, 0, 0]
            bbox_y = bnd_boxes[:, 0, 1]
            bbox_c_x = bnd_boxes[:, 0, 0] + (bnd_boxes[:, 1, 0]) / 2
            bbox_c_y = bnd_boxes[:, 0, 1] + (bnd_boxes[:, 1, 1]) / 2
            print(" bbox centres (x,y): ", bbox_c_x, bbox_c_y)


            x_rej = (bbox_x > x_rej_thresh)
            # rej if (xmin < 10) or (ymin + height > 2150)
            x_edge_rej = (bbox_x < 10)
            y_edge_rej = (bbox_y + bbox_h > 2150)
            # print(" x_rej: ", x_rej, "  y_rej: ", y_rej)
            # rej_mask = ((x_rej & y_rej) | x_edge_rej | y_edge_rej)
            rej_mask = (x_rej | x_edge_rej | y_edge_rej)

            print(" x_thresh: ", x_rej_thresh, "  y_thresh: ", y_rej_thresh)
            print(bbox_x > x_rej_thresh)
            print(" x rej: ", x_rej)
            print(" x edge rej: ", x_edge_rej, "  y edge rej: ", y_edge_rej)
            print(" xy rej_mask: ", rej_mask, np.where(rej_mask)[0].tolist())
            print(len(frame_detections[frame_name]))
            print(len([frame_detections[frame_name][j] for j in range(len(frame_detections[frame_name])) 
                                        if j not in rej_mask]))

            # just handle 2 for now, hope for the best...
            if len(bbox_x) == 2:
                centre_dist = np.sqrt((bbox_c_x[0] - bbox_c_x[1])**2 + (bbox_c_y[0] - bbox_c_y[1])**2)
                # print(" centre_dist: ", centre_dist)
                if centre_dist < bbox_centre_dist_thresh:
                    frame_detections[frame_name] = [frame_detections[frame_name][0]]

        # Fill empty detection if needed
        if frame_detections[frame_name] == []:
            frame_detections[frame_name] = [{}]
        # print("Kept (", len(frame_detections[frame_name]), ") :", frame_detections[frame_name])

        assert len(frame_detections[frame_name]) != 0, "must have proper \"empty\" detection in frame!"
        assert len(frame_detections[frame_name]) == 1, "Kept more than one detection in frame!"

    # ungroup detections from per-frame format back to original format
    filtered_alphapose_results = [frame_detections[frame_name][0] for frame_name in frame_names]

    return filtered_alphapose_results    

# TODO: REPLACE PLACEHOLDER WITH REAL LOADING ONCE DATASET IS SETUP
def get_2D_keypoints_from_alphapose_dict(data_path, cam_idx, subj, task, 
                                         kpts_dict=None, norm_cam=False):
    '''
    Loads the subject 2D keypoints from the alphapose json outputs
    '''
    NUM_ALPHAPOSE_KPTS = 26
    # Initialize the dictionary components as needed
    if kpts_dict is None:
        kpts_dict = {}
    if subj not in kpts_dict:
        kpts_dict[subj] = {}
    if task not in kpts_dict[subj]:
        kpts_dict[subj][task] = {'pos': {}, 'conf': {}}

    if cam_idx in kpts_dict[subj][task]['pos'] or cam_idx in kpts_dict[subj][task]['conf']:
        print('  ERR: Camera indexed data already exists in dict!')
        return kpts_dict

    # Get in there
    with open(data_path) as f:
        alphapose_results = json.load(f)
    
    # Filter out the detections that are for wrong person or duplicated
    alphapose_results = filter_alphapose_preds(alphapose_results, cam_idx)

    # TODO: HOW BEST TO HANDLE EMPTY FRAMES?
    # remove frames with invalid predictions:
    alphapose_results = [alphapose_results[i] for i in range(len(alphapose_results)) if alphapose_results[i] != {}]
    # nothing to do if there are no valid detections in this video
    if len(alphapose_results) == 0:
        return kpts_dict

    # Convert to numpy arrays
    keypoints = np.array([np.array(alphapose_results[i]["keypoints"]).reshape(-1,3) for i in range(len(alphapose_results))])    

    # keypoints entries have 3 values: x, y, confidence. And we only want 15 of the Halpe-26 keypoints
    kpts = keypoints[:, :, :2][:, 5:20]  # (frames, ktps, xy)
    conf = keypoints[:, :, 2][:, 5:20]  # (frames, kpts, conf)

    # Normalize the keypoints
    # TODO: HOW IS THIS SUPPOSED TO WORK? I THINK IT SHOULD NORM OVER ALL FRAMES FOR A CHANNEL?
    if norm_cam:
        kpts = kpts - kpts[:, -1:, :]    # TODO: ZERO TO first frame?
        kpts /= np.linalg.norm(kpts, ord=2, axis=1, keepdims=True)   # TODO: WHAT AXIS SHOULD BE NORMED?

    # Store the keypoints
    kpts_dict[subj][task]['pos'][cam_idx] = kpts.reshape((kpts.shape[0], -1), order='F')
    kpts_dict[subj][task]['conf'][cam_idx] = conf.reshape((kpts.shape[0], -1), order='F')

    return kpts_dict

# For loading the extracted 3D body keypoints from the UPDRS dataset
class body_ts_loader():
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
    

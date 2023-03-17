from utils import info
from scipy.signal import savgol_filter
import numpy as np
import json


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
    img_w = 3840 //2    # TODO: WHY DO THE DETECTION COORDS NOT SEEM TO LINE UP WITH THE IMG DIMS? IS ALPHAPOSE RESIZING?
    img_h = 2160 //2
    if cam_idx == '003':
        x_rej_thresh = 1/2 * img_w
        y_rej_thresh = 3/4 * img_h
    elif cam_idx == '004':
        x_rej_thresh = 3/4 * img_w
        y_rej_thresh = 2/3 * img_h

    score_rej_thresh = 1.75  # TODO: DETERMINE A GOOD THRESHOLD TO USE?
    bbox_centre_dist_thresh = 150  # TODO: DETERMINE A GOOD THRESHOLD TO USE?

    # choose the detection to keep for each frame, if any
    for i, frame_name in enumerate(frame_names):
        # get the bounding boxes for the detections in this frame
        bnd_boxes = np.array([np.array(frame_detections[frame_name][j]["box"]).reshape(-1,2) for j in range(len(frame_detections[frame_name]))])
        # get the centre(s) of the bounding box(es)
        bbox_x = bnd_boxes[:, 0, 0] + (bnd_boxes[:, 1, 0] - bnd_boxes[:, 0, 0]) / 2
        bbox_y = bnd_boxes[:, 0, 1] + (bnd_boxes[:, 1, 1] - bnd_boxes[:, 0, 1]) / 2

        # print("\nrej thresh: ", x_rej_thresh, y_rej_thresh,"     bbox centres (x,y): ", bbox_x, bbox_y)
        mags = np.sqrt(bbox_x**2 + bbox_y**2)
        # print(" bbx mags: ", mags)
        
        # create frame position rejection mask
        x_rej = (bbox_x > x_rej_thresh)
        y_rej = (bbox_y < y_rej_thresh)
        # print(" x_rej: ", x_rej, "  y_rej: ", y_rej)
        rej_mask = (x_rej & y_rej)
        # print(" xy rej_mask: ", rej_mask)
        rej_mask = np.where(rej_mask)[0].tolist()
        # print(" xy rej_mask idxs: ", rej_mask)

        # reject detections
        frame_detections[frame_name] = [frame_detections[frame_name][j] for j in range(len(frame_detections[frame_name])) 
                                        if j not in rej_mask]
        # also reject if score is too low
        frame_detections[frame_name] = [frame_detections[frame_name][j] for j in range(len(frame_detections[frame_name])) 
                                        if frame_detections[frame_name][j]["score"] > score_rej_thresh]
        
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
            idxs = np.array([np.array(frame_detections[frame_name][j]["idx"]) for j in range(len(frame_detections[frame_name]))])
            # get the centre(s) of the bounding box(es)
            bbox_x = bnd_boxes[:, 0, 0] + (bnd_boxes[:, 1, 0] - bnd_boxes[:, 0, 0]) / 2
            bbox_y = bnd_boxes[:, 0, 1] + (bnd_boxes[:, 1, 1] - bnd_boxes[:, 0, 1]) / 2

            # just handle 2 for now, hope for the best...
            if len(bbox_x) == 2:
                centre_dist = np.sqrt((bbox_x[0] - bbox_x[1])**2 + (bbox_y[0] - bbox_y[1])**2)
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
                                         kpts_dict=None, norm_cam=True):
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
    

from utils import info
from scipy.signal import savgol_filter
import numpy as np
import json

# For loading the raw subject 2D keypoints from the alphapose json outputs
# TODO: REPLACE PLACEHOLDER WITH REAL LOADING ONCE DATASET IS SETUP
def get_2D_keypoints_dict(data_path, tasks, channels, norm_cam=False):
    # TODO: HANDLE FRAME SELECTION, CURRENTLY JUST GRABS FIRST FRAME
    #
    # Should load all frames for a given task, and then normalize over all frames for a given channel
    #
    keypoints_PD = {}
    for task in tasks:
        for subj in ['9769']: # TODO: MAP TO S01, S02, etc.
            keypoints_PD[subj] = {}
            keypoints_PD[subj][task] = {'pos': {}, 'conf': {}}
            for cam_idx in channels:
                with open(data_path + subj + "/" + task + '_CH' + str(cam_idx) + '/alphapose-results.json') as f:
                    alphapose_results = json.load(f)
                
                # For now, just grab first entry for each frame as we always have 2 people in the frame
                # TODO: DETERMINE HOW TO HANDLE MULTIPLE PEOPLE MORE ROBUSTLY
                num_frames = len(alphapose_results) // 2
                idxs = [int(fr*2) for fr in range(num_frames)]  
                keypoints = np.array([np.array(alphapose_results[i]["keypoints"]).reshape(-1,3) for i in idxs])

                # keypoints entries have 3 values: x, y, confidence. And we only want 15 of the Halpe-26 keypoints
                kpts = keypoints[:, :, :2][:, 5:20]  # (frames, ktps, xy)
                conf = keypoints[:, :, 2][:, 5:20]  # (frames, kpts, conf)

                # Normalize the keypoints
                # TODO: HOW IS THIS SUPPOSED TO WORK? I THINK IT SHOULD NORM OVER ALL FRAMES FOR A CHANNEL
                if norm_cam:
                    kpts = kpts - kpts[:, -1:, :]    # TODO: WHAT COORD SHOULD I SUBTRACT? PELVIS I THINK?
                    kpts /= np.linalg.norm(kpts, ord=2, axis=1, keepdims=True)   # TODO: WHAT AXIS SHOULD BE NORMED?

                # Store the keypoints
                keypoints_PD[subj][task]['pos'][cam_idx] = kpts.reshape((kpts.shape[0], -1), order='F')
                keypoints_PD[subj][task]['conf'][cam_idx] = conf.reshape((kpts.shape[0], -1), order='F')
    return keypoints_PD

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
    

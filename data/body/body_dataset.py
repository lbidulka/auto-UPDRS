from utils import info
from scipy.signal import savgol_filter
import numpy as np
import json

# For loading the subject 2D keypoints from the alphapose json outputs
# TODO: REPLACE PLACEHOLDER WITH REAL LOADING ONCE DATASET IS SETUP
def get_2D_keypoints_dict(data_path, tasks, channels):
    keypoints_PD = {}
    for task in tasks:
        for subj in ['9769']: # TODO: MAP TO S01, S02, etc.
            keypoints_PD[subj] = {}
            keypoints_PD[subj][task] = {'pos': {}, 'conf': {}}
            for cam_idx in channels:
                with open(data_path + subj + "/" + task + '_CH' + str(cam_idx) + '/alphapose-results.json') as f:
                    alphapose_results = json.load(f)
                # keypoints entries have 3 values: x, y, confidence. And we only want 15 of the Halpe-26 keypoints
                keypoints = np.array(alphapose_results[0]["keypoints"]).reshape(-1,3)   # TODO: DETERMINE HOW TO HANDLE MULTIPLE PEOPLE
                keypoints_PD[subj][task]['pos'][cam_idx] = keypoints[:,:2][5:20].reshape(-1)  # xy
                keypoints_PD[subj][task]['conf'][cam_idx] = keypoints[:,2][5:20].reshape(-1)  # conf
    return keypoints_PD

# For loading the extracted 3D body keypoints from the UPDRS dataset
class body_ts_loader():
    def __init__(self, ts_path, subjects = info.subjects_All) -> None:
        self.subjects = subjects
        self.ts_path = ts_path
        self.feat_names = info.clinical_gait_feat_names

        self.data_normal = []
        for idx, subj in enumerate(subjects):
            data_normal = np.load(ts_path + 'Predictions_' + subj + '.npy')
            for ii in range(15):
                for jj in range(3):
                    data_normal[:,ii,jj] = savgol_filter(data_normal[:,ii,jj], 11, 3)   # Smoothing
            
            x_vec = data_normal[:,1] - data_normal[:,4]
            y_vec = data_normal[:,7] - data_normal[:,0]
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
    

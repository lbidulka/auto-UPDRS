import numpy as np
from scipy.signal import savgol_filter

class gait_features():
    def __init__(self, subjects, ts_path) -> None:
        self.subjects = subjects
        self.ts_path = ts_path

        self.data_normal = []

        for idx, subj in enumerate(subjects):
            # './Weakly_Supervised_Learning/outputs_finetuned/Predictions_'
            data_normal = np.load(ts_path + 'Predictions_' + subj + '.npy')
            for ii in range(15):
                for jj in range(3):
                    data_normal[:,ii,jj] = savgol_filter(data_normal[:,ii,jj], 11, 3)
            
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
    
    def step_width(self, subjects):

        step_widths = []
        for subj in subjects:
            idx = self.subjects.index(subj)
            step_width = np.abs(self.data_normal[idx][:,3,0] - self.data_normal[idx][:,6,0])
            bone_length = np.linalg.norm((self.data_normal[idx][:,1] - self.data_normal[idx][:,4]),axis=-1)
            step_width /= bone_length

            A=[]
            for ii in range(30, len(step_width)):
                A.append(np.mean(step_width[ii-30:ii]))

            step_widths.append(np.mean(np.asarray(A)))
        
        return step_widths
            

import numpy as np
from scipy.signal import savgol_filter, find_peaks
from utils import info
 
# TODO: DETERMINE IF MEAN OF FEATURES IS OVER TIME? IF SO, COULD WE USE RAW DATA?
class gait_processor():
    def __init__(self, ts_path, subjects = info.subjects_All) -> None:
        self.subjects = subjects
        self.ts_path = ts_path
        self.feat_names = info.clinical_gait_feat_names

        self.data_normal = []
        for idx, subj in enumerate(subjects):
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
        
        self.feats = self.compute_features(subjects)
        self.thresholds = self._set_thresholds()
    
    # Use thresholds to get indicators from subject features
    # NB: we may group them according to biomechanical correlation, in the order as in the paper (not Mohsens code)
    def compute_indicators(self, feats, grouped = False):
        mins = self.thresholds[0].reshape(-1,1)
        maxs = self.thresholds[1].reshape(-1,1)

        ungrouped_indicators = np.logical_and(mins < feats, feats < maxs) * 1
        ungrouped_indicators[14] = -ungrouped_indicators[14] + 1    # Need to logically invert this one

        if not grouped:
            return ungrouped_indicators

        indicators = np.zeros((8, feats.shape[1]))
        # Hand: R_arm_swing, L_arm_swing, arm_sym
        indicators[0] = np.amax(np.array([ungrouped_indicators[6], ungrouped_indicators[7], ungrouped_indicators[14]]), axis=0)
        # Step: step_len_R, step_len_L
        indicators[1] = np.amax(np.array([ungrouped_indicators[1], ungrouped_indicators[2]]), axis=0)
        # Foot: footlift_R, footlift_L
        indicators[2] = np.amax(np.array([ungrouped_indicators[4], ungrouped_indicators[5]]), axis=0)
        # Hip: hip_flex_R, hip_flex_L
        indicators[3] = np.amax(np.array([ungrouped_indicators[8], ungrouped_indicators[9]]), axis=0)
        # Knee: knee_flex_R, knee_flex_L
        indicators[4] = np.amax(np.array([ungrouped_indicators[10], ungrouped_indicators[11]]), axis=0)
        # Trunk: trunk_rot_R, trunk_rot_L
        indicators[5] = np.amax(np.array([ungrouped_indicators[12], ungrouped_indicators[13]]), axis=0)
        # Cadence, Step Width
        indicators[6] = ungrouped_indicators[3]
        indicators[7] = ungrouped_indicators[0]

        return indicators
    
    # Computed using all healthy controls.
    #
    # if x falls within the range of the thresholds, then labeling function outputs 1, else 0
    #
    # logic:
    #
    #   step_width:  x > max   # TODO: CHECK THIS, IN PAPER IT SAID ONLY CADENCE WAS MAX FUNCTION???
    #   step_len_R:  x < min
    #   step_len_L:  x < min
    #   cadence:      x > max   
    #   footlift_R:  x < min
    #   footlift_L:  x < min
    #   R_arm_swing: x < min
    #   L_arm_swing: x < min
    #   hip_flex_R:  x < min
    #   hip_flex_L:  x < min
    #   knee_flex_R: x < min
    #   knee_flex_L: x < min
    #   trunk_rot_R: x < min
    #   trunk_rot_L: x < min
    #   arm_sym:     (x < min) or (max < x)
    #
    #   gait_speed: x > max   # TODO: CHECK IF THIS IS ACTUALLY USED???
    #
    def _set_thresholds(self):

        control_feats = self.compute_features(info.healthy_controls)

        # Handle most of them
        thresholds_min = np.ones(len(info.clinical_gait_feat_names)) * -np.inf
        thresholds_max = np.amin(control_feats[:-2], axis=1) # Dont include gait speed and gait speed var.
        # Step Width, Cadence
        thresholds_min[0] = np.amax(control_feats[0])
        thresholds_max[0] = np.inf
        thresholds_min[3] = np.amax(control_feats[3])
        thresholds_max[3] = np.inf
        # Arm swing symmetry
        thresholds_min[14] = np.amin(control_feats[14])    # TODO: FIX LOGIC, REVERSE THE MIN AND MAX?
        thresholds_max[14] = np.amax(control_feats[14])    
        
        return np.vstack([thresholds_min, thresholds_max])
    
    def compute_features(self, subjects):
        step_widths = np.array(self._step_width(subjects))
        step_lengths = np.array(self._step_lengths(subjects))
        cadences_gaitspeeds_gaitspeedvars = np.array(self._cadence_gaitspeed_gaitspeedvar(subjects))
        foot_lifts = np.array(self._foot_lifts(subjects))
        arm_swings = np.array(self._arm_swings(subjects))
        hip_flexions = np.array(self._hip_flexions(subjects))
        knee_flexions = np.array(self._knee_flexions(subjects))
        trunk_rots = np.array(self._trunk_rots(subjects))

        return np.vstack([
            step_widths,
            step_lengths[:, 0],  # R Step Length
            step_lengths[:, 1],   # L Step Length
            cadences_gaitspeeds_gaitspeedvars[:, 0],  # Cadence
            foot_lifts[:, 0],  # R Foot Clearance
            foot_lifts[:, 1],  # L Foot Clearance
            arm_swings[:, 0],  # R Arm Swing
            arm_swings[:, 1],  # L Arm Swing
            hip_flexions[:, 0],  # R Hip Flexion
            hip_flexions[:, 1],  # L Hip Flexion
            knee_flexions[:, 0],  # R Knee Flexion
            knee_flexions[:, 1],  # L Knee Flexion
            trunk_rots[:, 0],  # R Trunk rotation
            trunk_rots[:, 1],  # L Trunk rotation
            arm_swings[:, 2],  # Arm swing symmetry
            cadences_gaitspeeds_gaitspeedvars[:, 1],  # Gait speed
            cadences_gaitspeeds_gaitspeedvars[:, 2],  # Gait Speed var
        ])
    
    def _step_width(self, subjects):
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
    
    def _step_lengths(self, subjects):
        step_lengths = []
        for subj in subjects:
            idx = self.subjects.index(subj)
            step_length = np.linalg.norm((self.data_normal[idx][:,3] - self.data_normal[idx][:,6]), axis=-1)
            row_r = (self.data_normal[idx][:,6,2] - self.data_normal[idx][:,3,2] > 0)
            row_l = (self.data_normal[idx][:,3,2] - self.data_normal[idx][:,6,2] > 0)
            step_length_r = step_length[row_r]
            step_length_l = step_length[row_l]
            bone_length_r = np.linalg.norm((self.data_normal[idx][:,5] - self.data_normal[idx][:,6]), axis=-1)[row_r]
            bone_length_l = np.linalg.norm((self.data_normal[idx][:,3] - self.data_normal[idx][:,2]), axis=-1)[row_l]
            step_length_r /= bone_length_r
            step_length_l /= bone_length_l

            A=[]
            for ii in range(30,len(step_length_r)):
                A.append(np.max(step_length_r[ii-30:ii]))
            step_length_r = np.asarray(A)
            B=[]
            for ii in range(30,len(step_length_l)):
                B.append(np.max(step_length_l[ii-30:ii]))
            step_length_l = np.asarray(B)

            step_lengths.append([np.mean(step_length_r), np.mean(step_length_l)])
        
        return step_lengths
            
    def _cadence_gaitspeed_gaitspeedvar(self, subjects):
        cadences_and_speeds = []
        for subj in subjects:
            idx = self.subjects.index(subj)
            toe_traj = np.abs(self.data_normal[idx][:,6,2] - self.data_normal[idx][:,3,2])
            peaks, _ = find_peaks(toe_traj, distance=5, height=-0.2)
            ave = np.mean(toe_traj[peaks]) - 0.3
            peaks, _ = find_peaks(toe_traj, distance=5, height=ave)

            # TODO: FIND OUT WHY THIS IS DIFFERENT FOR S01, S28, S29, S31
            if subj in ['S01','S28','S29','S31']:
                cadence = 60/((peaks[1:]-peaks[:-1])/30)
                gait_speed = toe_traj[peaks[1:]] * cadence
            else:
                cadence = 60 / ((peaks[1:] - peaks[:-1]) / 15)
                gait_speed = toe_traj[peaks[1:]] * cadence
        
            cadences_and_speeds.append([np.mean(cadence), np.mean(gait_speed), np.std(gait_speed)])

        return cadences_and_speeds
    
    def _foot_lifts(self, subjects):
        foot_lifts = []
        for subj in subjects:
            idx = self.subjects.index(subj)

            foot_height_r = self.data_normal[idx][:,6,1] - self.data_normal[idx][:,4,1]
            foot_height_l = self.data_normal[idx][:,3,1] - self.data_normal[idx][:,1,1]
            bone_length_l = np.linalg.norm((self.data_normal[idx][:,1] - self.data_normal[idx][:,2]), axis=-1)
            bone_length_r = np.linalg.norm((self.data_normal[idx][:,5] - self.data_normal[idx][:,4]), axis=-1)
            foot_height_r /= bone_length_r
            foot_height_l /= bone_length_l

            A=[]
            B=[]
            for ii in range(30,len(foot_height_r)):
                A.append(np.max(foot_height_r[ii-30:ii]) - np.min(foot_height_r[ii-30:ii]))
                B.append(np.max(foot_height_l[ii-30:ii]) - np.min(foot_height_l[ii-30:ii]))
            foot_height_n_r = np.asarray(A)
            foot_height_n_l = np.asarray(B)

            foot_lifts.append([np.mean(foot_height_n_r), np.mean(foot_height_n_l)])
        
        return foot_lifts

    def _arm_swings(self, subjects):
        arm_swings = []
        for subj in subjects:
            idx = self.subjects.index(subj)

            # Right hand
            dist = np.linalg.norm((self.data_normal[idx][:,14] - self.data_normal[idx][:,4]), axis=-1)
            bone_length = np.linalg.norm((self.data_normal[idx][:,4] - self.data_normal[idx][:,1]), axis=-1)
            dist /= bone_length

            A=[]
            for ii in range(30,len(dist)):
                A.append(np.max(dist[ii-30:ii]) - np.min(dist[ii-30:ii]))
            hand_mov_n_r = np.asarray(A)

            # Left hand
            dist = np.linalg.norm((self.data_normal[idx][:,11] - self.data_normal[idx][:,1]), axis=-1)
            dist /= bone_length

            A=[]
            for ii in range(30,len(dist)):
                A.append(np.max(dist[ii-30:ii]) - np.min(dist[ii-30:ii]))
            hand_mov_n_l = np.asarray(A)

            arm_swings.append([np.mean(hand_mov_n_r), np.mean(hand_mov_n_l), 
                                   np.mean(hand_mov_n_l) / np.mean(hand_mov_n_r)])
        
        return arm_swings
    
    def _hip_flexions(self, subjects):
        hip_flexions = []
        for subj in subjects:
            idx = self.subjects.index(subj)

            dist_l = self.data_normal[idx][:,1,2] - self.data_normal[idx][:,2,2]
            bone_l = np.linalg.norm((self.data_normal[idx][:,1] - self.data_normal[idx][:,2]), axis=-1)
            dist_r = self.data_normal[idx][:,4,2] - self.data_normal[idx][:,5,2]
            bone_r = np.linalg.norm((self.data_normal[idx][:,4] - self.data_normal[idx][:,5]), axis=-1)
            hip_flex_r = dist_r / bone_r
            hip_flex_l = dist_l / bone_l

            A = []
            B = []
            for ii in range(30,len(hip_flex_r)):
                A.append(np.max(hip_flex_r[ii-30:ii]) - np.min(hip_flex_r[ii-30:ii]))
                B.append(np.max(hip_flex_l[ii-30:ii]) - np.min(hip_flex_l[ii-30:ii])) 
            hip_flex_r = np.asarray(A)
            hip_flex_l = np.asarray(B)

            hip_flexions.append([np.mean(hip_flex_r), np.mean(hip_flex_l)])

        return hip_flexions

    def _knee_flexions(self, subjects):
        knee_flexions = []
        for subj in subjects:
            idx = self.subjects.index(subj)

            thigh_l = np.linalg.norm((self.data_normal[idx][:,1] - self.data_normal[idx][:,2]), axis=-1)
            shin_l = np.linalg.norm((self.data_normal[idx][:,3] - self.data_normal[idx][:,2]), axis=-1)
            leg_l = np.linalg.norm((self.data_normal[idx][:,1] - self.data_normal[idx][:,3]), axis=-1)
            thigh_r = np.linalg.norm((self.data_normal[idx][:,5] - self.data_normal[idx][:,4]), axis=-1)
            shin_r = np.linalg.norm((self.data_normal[idx][:,5] - self.data_normal[idx][:,6]), axis=-1)
            leg_r = np.linalg.norm((self.data_normal[idx][:,4] - self.data_normal[idx][:,6]), axis=-1)
            knee_flex_r = leg_r**2 / (thigh_r * shin_r) - thigh_r / shin_r - shin_r / thigh_r
            knee_flex_l = leg_l**2 / (thigh_l * shin_l) - thigh_l / shin_l - shin_l / thigh_l

            A = []
            B = []
            for ii in range(30,len(knee_flex_r)):
                A.append(np.max(knee_flex_r[ii-30:ii]) - np.min(knee_flex_r[ii-30:ii]))
                B.append(np.max(knee_flex_l[ii-30:ii]) - np.min(knee_flex_l[ii-30:ii]))
            knee_flex_r = np.asarray(A)
            knee_flex_l = np.asarray(B)

            # TODO: IS THIS THE RIGHT ORDER? MOHSEN HAD IT SWAPPED (L, R), OPPOSITE TO ALL OTHERS
            knee_flexions.append([np.mean(knee_flex_r), np.mean(knee_flex_l)])
            
        return knee_flexions

    def _trunk_rots(self, subjects):
        trunk_rots = []
        for subj in subjects:
            idx = self.subjects.index(subj)

            data_normal = self.data_normal[idx] - self.data_normal[idx][:,:1]
            shoulder_l = np.linalg.norm((data_normal[:,9,[0,2]] - data_normal[:,0,[0,2]]), axis=-1)
            hip_l = np.linalg.norm((data_normal[:,4,[0,2]] - data_normal[:,0,[0,2]]), axis=-1)
            hip2shoulder_l = np.linalg.norm((data_normal[:,4,[0,2]] - data_normal[:,9,[0,2]]), axis=-1)
            shoulder_r = np.linalg.norm((data_normal[:,12,[0,2]] - data_normal[:,0,[0,2]]), axis=-1)
            hip_r = np.linalg.norm((data_normal[:,1,[0,2]] - data_normal[:,0,[0,2]]), axis=-1)
            hip2shoulder_r = np.linalg.norm((data_normal[:,1,[0,2]] - data_normal[:,12,[0,2]]), axis=-1)
            trunk_rot_r = hip2shoulder_r**2 / (hip_r * shoulder_r) - hip_r / shoulder_r - shoulder_r / hip_r
            trunk_rot_l = hip2shoulder_l**2 / (hip_l * shoulder_l) - hip_l / shoulder_l - shoulder_l / hip_l

            A = []
            B = []
            for ii in range(30,len(trunk_rot_r)):
                A.append(np.max(trunk_rot_r[ii-30:ii]) - np.min(trunk_rot_r[ii-30:ii]))
                B.append(np.max(trunk_rot_l[ii-30:ii]) - np.min(trunk_rot_l[ii-30:ii]))   
            trunk_rot_r = np.asarray(A)
            trunk_rot_l = np.asarray(B)

            trunk_rots.append([np.mean(trunk_rot_r), np.mean(trunk_rot_l)])
            
        return trunk_rots

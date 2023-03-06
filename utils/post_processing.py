import numpy as np
from scipy.signal import savgol_filter, find_peaks
from utils import info
from data.body.time_series.body_dataset import body_ts_loader
import matplotlib.pyplot as plt

# For processing extracted 3D body keypoints 
class gait_processor():
    def __init__(self, body_ts_loader, fig_outpath) -> None:
        self.fig_outpath = fig_outpath
        self.subjects = body_ts_loader.subjects
        self.data_normal = body_ts_loader.data_normal
        self.feat_names = info.clinical_gait_feat_names
        self.feats_avg = self.compute_features(self.subjects, ts=False)
        self.feats_ts = self.compute_features(self.subjects, ts=True)
        self.thresholds = self._set_thresholds()
    
    # Plot the time series of the features
    #
    # Notes:
    #   - Some subjects have longer time series than others
    #   - Cadence is not properly the timeseries yet, it is based on peaks
    # TODO: CREATE PROPER CADENCE TIMESERIES
    def plot_feats_ts(self, save_fig=True):
        fig_rows = 5 #5
        fig_cols = 3 #3
        fig, ax = plt.subplots(fig_rows, fig_cols, layout="constrained")
        fig.set_size_inches(18.5, 10.5)
        # plot
        for ii in range(len(info.clinical_gait_feat_names)): # 10):
            for jj in range(5, 10): #range(len(self.subjects)):
                ax[ii//fig_cols, ii%fig_cols].plot(self.feats_ts[ii][jj], linewidth=1, alpha=0.5, 
                                                   label=self.subjects[jj] if ii==0 else None)
            ax[ii//fig_cols, ii%fig_cols].set_title(info.clinical_gait_feat_names[ii])
            ax[ii//fig_cols, ii%fig_cols].set_xlabel('Timestep')
        fig.legend(loc='right', bbox_to_anchor=(1, 0.5))
        if save_fig: plt.savefig(self.fig_outpath + 'feats_ts.png', dpi=500)
        plt.show()

    def plot_preds_by_feats(self, feats, y_preds, thresholds=True):
        fig_rows = 5
        fig_cols = 3
        fig, ax = plt.subplots(fig_rows, fig_cols, layout="tight")
        # plot
        for ii in range(len(info.clinical_gait_feat_names)):
            ax[ii//fig_cols, ii%fig_cols].scatter(feats[ii,:], y_preds[ii,:])
            ax[ii//fig_cols, ii%fig_cols].set_title(info.clinical_gait_feat_names[ii])
            ax[ii//fig_cols, ii%fig_cols].set_xlabel('Feat Val')
            ax[ii//fig_cols, ii%fig_cols].set_ylabel('UPDRS > 0')
        # if thresholds, add the threshold lines
        if thresholds:
            for ii in range(len(info.clinical_gait_feat_names)):
                ax[ii//fig_cols, ii%fig_cols].axvline(self.thresholds[0][ii], color='b')    # min threshold
                ax[ii//fig_cols, ii%fig_cols].axvline(self.thresholds[1][ii], color='r')    # max threshold
        plt.show()
    
    def plot_feats(self, subjects, thresholds=True):
        fig_rows = 5
        fig_cols = 3
        fig, ax = plt.subplots(fig_rows, fig_cols, layout="tight")
        # y vals are 0 for healthy controls, 1 for PD
        y_vals = np.ones(len(subjects))
        for ii, subj in enumerate(subjects):
            if subj in info.healthy_controls:
                y_vals[ii] = 0
        control_feats = self.compute_features(info.healthy_controls, avg=True)
        pd_feats = self.compute_features(info.subjects_PD, avg=True)
        # plot
        for ii in range(len(info.clinical_gait_feat_names)):
            ax[ii//fig_cols, ii%fig_cols].scatter(control_feats[ii,:], np.zeros((len(control_feats[ii,:]), 1)))
            ax[ii//fig_cols, ii%fig_cols].scatter(pd_feats[ii,:], np.ones((len(pd_feats[ii,:]), 1)))
            ax[ii//fig_cols, ii%fig_cols].set_title(info.clinical_gait_feat_names[ii])
            ax[ii//fig_cols, ii%fig_cols].set_xlabel('Feat Val')
            ax[ii//fig_cols, ii%fig_cols].set_ylabel('UPDRS > 0')
        # if thresholds, add the threshold lines
        if thresholds:
            for ii in range(len(info.clinical_gait_feat_names)):
                ax[ii//fig_cols, ii%fig_cols].axvline(self.thresholds[0][ii], color='b')    # min threshold
                ax[ii//fig_cols, ii%fig_cols].axvline(self.thresholds[1][ii], color='r')    # max threshold
        plt.show()

    # Use thresholds to get indicators from subject features
    # NB: we may group them according to biomechanical correlation, in the order as in the paper (not Mohsens code)
    def compute_indicators(self, feats, grouped = False):
        mins = self.thresholds[0].reshape(-1,1)
        maxs = self.thresholds[1].reshape(-1,1)
        ungrouped_indicators = np.logical_and(mins < feats, feats < maxs) * 1
        ungrouped_indicators[14] = -ungrouped_indicators[14] + 1    # Need to logically invert this one
        if not grouped:
            return ungrouped_indicators
        return self._group_indicators(ungrouped_indicators)
    
    # Group the ungrouped indicators
    def _group_indicators(self, ungrouped_indicators):
        indicators = np.zeros((8, ungrouped_indicators.shape[1]))
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
    # if (thresh_min < x < thresh_max): 1
    # else: 0
    #
    # logic, w.r.t data values:
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
        control_feats = self.compute_features(info.healthy_controls, ts=False)
        # Handle most of them
        thresholds_min = np.ones(len(info.clinical_gait_feat_names)) * -np.inf
        thresholds_max = np.amin(control_feats[:15], axis=1) # Dont include gait speed and gait speed var.
        # Step Width, Cadence
        thresholds_min[0] = np.amax(control_feats[0])
        thresholds_max[0] = np.inf
        thresholds_min[3] = np.amax(control_feats[3])
        thresholds_max[3] = np.inf
        # Arm swing symmetry
        thresholds_min[14] = np.amin(control_feats[14])    # TODO: FIX LOGIC, REVERSE THE MIN AND MAX?
        thresholds_max[14] = np.amax(control_feats[14])    
        return np.vstack([thresholds_min, thresholds_max])
    
    def compute_features(self, subjects, ts = False):
        if ts:
            step_widths = self._step_width(subjects, ts)
            step_lengths = self._step_lengths(subjects, ts)
            cadences_gaitspeeds_gaitspeedvars = self._cadence_gaitspeed_gaitspeedvar(subjects, ts)
            foot_lifts = self._foot_lifts(subjects, ts)
            arm_swings = self._arm_swings(subjects, ts)
            hip_flexions = self._hip_flexions(subjects, ts)
            knee_flexions = self._knee_flexions(subjects, ts)
            trunk_rots = self._trunk_rots(subjects, ts)
            return [
                # avg of L and R step lengths for each timestep
                step_widths,
                [lens[0, :] for lens in step_lengths], # R Step Length
                [lens[1, :] for lens in step_lengths], # L Step Length
                [cadences[0] for cadences in cadences_gaitspeeds_gaitspeedvars], # Cadence
                [lifts[0] for lifts in foot_lifts], # R Foot Clearance
                [lifts[1] for lifts in foot_lifts], # L Foot Clearance
                [swings[0] for swings in arm_swings], # R Arm Swing
                [swings[1] for swings in arm_swings], # L Arm Swing
                [flexions[0] for flexions in hip_flexions], # R Hip Flexion
                [flexions[1] for flexions in hip_flexions], # L Hip Flexion
                [flexions[0] for flexions in knee_flexions], # R Knee Flexion
                [flexions[1] for flexions in knee_flexions], # L Knee Flexion
                [rots[0] for rots in trunk_rots], # R Trunk Rotation
                [rots[1] for rots in trunk_rots], # L Trunk Rotation
                [swings[2] for swings in arm_swings], # Arm Swing Symmetry
                [cadences[1] for cadences in cadences_gaitspeeds_gaitspeedvars], # Gait Speed
            ]
        
        step_widths = np.array(self._step_width(subjects, ts))
        step_lengths = np.array(self._step_lengths(subjects, ts))
        cadences_gaitspeeds_gaitspeedvars = np.array(self._cadence_gaitspeed_gaitspeedvar(subjects, ts))
        foot_lifts = np.array(self._foot_lifts(subjects, ts))
        arm_swings = np.array(self._arm_swings(subjects, ts))
        hip_flexions = np.array(self._hip_flexions(subjects, ts))
        knee_flexions = np.array(self._knee_flexions(subjects, ts))
        trunk_rots = np.array(self._trunk_rots(subjects, ts))
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
            # cadences_gaitspeeds_gaitspeedvars[:, 2],  # Gait Speed var
        ])

    # Helper to apply filtering to time series data
    def _filter_1d(self, data, filter=None, win_len=2, ord=3):
        if filter is None:
            return data
        if filter == 'moving_avg':
            return savgol_filter(data, win_len, 1)
        elif filter == 'savgol':
            return savgol_filter(data, win_len, ord)
    
    def _step_width(self, subjects, ts = False, filter="moving_avg"):
        step_widths = []
        for subj in subjects:
            idx = self.subjects.index(subj)
            step_width = np.abs(self.data_normal[idx][:,3,0] - self.data_normal[idx][:,6,0])
            bone_length = np.linalg.norm((self.data_normal[idx][:,1] - self.data_normal[idx][:,4]),axis=-1)
            step_width /= bone_length
            step_widths.append(self._filter_1d(step_width, filter) if ts else 
                                np.mean(self._filter_1d(step_width, "moving_avg", 30, 1)))
        return step_widths
    
    def _step_lengths(self, subjects, ts=False, filter=True):
        step_lengths = []
        for subj in subjects:
            idx = self.subjects.index(subj)
            step_length = np.linalg.norm((self.data_normal[idx][:,3] - self.data_normal[idx][:,6]), axis=-1)
            # IF RIGHT FOOT IS IN FRONT OF LEFT FOOT, & VICE VERSA
            row_r = (self.data_normal[idx][:,6,2] - self.data_normal[idx][:,3,2] > 0)
            row_l = (self.data_normal[idx][:,3,2] - self.data_normal[idx][:,6,2] > 0)
            step_length_r = step_length[row_r]
            step_length_l = step_length[row_l]
            #
            bone_length_r = np.linalg.norm((self.data_normal[idx][:,5] - self.data_normal[idx][:,6]), axis=-1)[row_r]
            bone_length_l = np.linalg.norm((self.data_normal[idx][:,3] - self.data_normal[idx][:,2]), axis=-1)[row_l]
            step_length_r /= bone_length_r
            step_length_l /= bone_length_l

            if not ts:
                # THIS -30 LEADS TO 60 LESS TIMESTEPS, WHAT IS THIS DOING?
                # TODO: WHY DO WE USE MAX FROM LAST 30 TIMESTEPS?
                A=[]
                for ii in range(30,len(step_length_r)):
                    A.append(np.max(step_length_r[ii-30:ii]))
                step_length_r = np.asarray(A)
                B=[]
                for ii in range(30,len(step_length_l)):
                    B.append(np.max(step_length_l[ii-30:ii]))
                step_length_l = np.asarray(B)

                step_lengths.append([np.mean(step_length_r), np.mean(step_length_l)])
            else:
                # Want timeseries of step lengths, vals update when corresponding foot is ahead (using row_r, row_l)
                sl_R = np.zeros(len(step_length))
                sl_L = np.zeros(len(step_length))
                sl_R[row_r] = step_length_r
                sl_L[row_l] = step_length_l
                # turn 0's into values to left, to hold val from last time foot was ahead
                for ii in range(1, len(sl_R)):
                    if sl_R[ii] == 0:
                        sl_R[ii] = sl_R[ii-1]
                for ii in range(1, len(sl_L)):
                    if sl_L[ii] == 0:
                        sl_L[ii] = sl_L[ii-1]
                if not filter:
                    step_lengths.append(np.vstack([sl_R, sl_L]))
                    break
                # take max of last 30 timesteps, to get peak step lengths
                # TODO: FIND OUT WHY THIS IS NEEDED, WHY IS THERE THE OSCILLATION IN RAW?
                sl_R_peaks = np.full_like(sl_R, 0)
                sl_L_peaks = np.full_like(sl_L, 0)
                for ii in range(30,len(sl_R)):
                    sl_R_peaks[ii] = np.max(sl_R[ii-30:ii])
                for ii in range(30,len(sl_L)):
                    sl_L_peaks[ii] = np.max(sl_L[ii-30:ii])
                step_lengths.append(np.vstack([sl_R_peaks, sl_L_peaks]))
        return step_lengths
            
    def _cadence_gaitspeed_gaitspeedvar(self, subjects, ts = False):
        cadences_and_speeds = []
        for subj in subjects:
            idx = self.subjects.index(subj)
            toe_traj = np.abs(self.data_normal[idx][:,6,2] - self.data_normal[idx][:,3,2])
            peaks, _ = find_peaks(toe_traj, distance=5, height=-0.2)
            ave = np.mean(toe_traj[peaks]) - 0.3
            peaks, _ = find_peaks(toe_traj, distance=5, height=ave)
            # TODO: FIND OUT WHY THIS IS DIFFERENT FOR S01, S28, S29, S31 -> DIFF FPS?
            if subj in ['S01','S28','S29','S31']:
                cadence = 60/((peaks[1:]-peaks[:-1])/30)
                gait_speed = toe_traj[peaks[1:]] * cadence
            else:
                cadence = 60 / ((peaks[1:] - peaks[:-1]) / 15)
                gait_speed = toe_traj[peaks[1:]] * cadence
            # TODO: MAP THESE VALUES TO THE TIMESTEPS OF THE DATA
            cadences_and_speeds.append([cadence, gait_speed] if ts else [np.mean(cadence), np.mean(gait_speed)])
        return cadences_and_speeds
    
    def _foot_lifts(self, subjects, ts = False, filter="moving_avg"):
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
            # TODO: DISSECT THIS COMPUTATION, THE MIN SHIFTS AS TIME GOES ON?
            for ii in range(30,len(foot_height_r)):
                A.append(np.max(foot_height_r[ii-30:ii]) - np.min(foot_height_r[ii-30:ii]))
                B.append(np.max(foot_height_l[ii-30:ii]) - np.min(foot_height_l[ii-30:ii]))
            foot_height_n_r = np.asarray(A)
            foot_height_n_l = np.asarray(B)
            foot_lifts.append([self._filter_1d(foot_height_n_r, filter), self._filter_1d(foot_height_n_l, filter)] if ts else 
                              [np.mean(foot_height_n_r), np.mean(foot_height_n_l)])   
        return foot_lifts

    def _arm_swings(self, subjects, ts = False, filter="moving_avg"):
        arm_swings = []
        for subj in subjects:
            idx = self.subjects.index(subj)
            bone_length = np.linalg.norm((self.data_normal[idx][:,4] - self.data_normal[idx][:,1]), axis=-1)
            dist_R = np.linalg.norm((self.data_normal[idx][:,14] - self.data_normal[idx][:,4]), axis=-1)
            dist_L = np.linalg.norm((self.data_normal[idx][:,11] - self.data_normal[idx][:,1]), axis=-1)
            dist_R /= bone_length
            dist_L /= bone_length
            dist_R = self._filter_1d(dist_R, filter)
            dist_L = self._filter_1d(dist_L, filter)
            arm_swings.append([dist_R, dist_L, dist_L / dist_R] if ts else
                                [np.mean(dist_R), np.mean(dist_L), np.mean(dist_L) / np.mean(dist_R)])
        return arm_swings
    
    def _hip_flexions(self, subjects, ts = False):
        hip_flexions = []
        for subj in subjects:
            idx = self.subjects.index(subj)
            dist_l = self.data_normal[idx][:,1,2] - self.data_normal[idx][:,2,2]
            bone_l = np.linalg.norm((self.data_normal[idx][:,1] - self.data_normal[idx][:,2]), axis=-1)
            bone_r = np.linalg.norm((self.data_normal[idx][:,4] - self.data_normal[idx][:,5]), axis=-1)
            dist_r = self.data_normal[idx][:,4,2] - self.data_normal[idx][:,5,2]
            hip_flex_r = dist_r / bone_r
            hip_flex_l = dist_l / bone_l
            A = []
            B = []
            for ii in range(30,len(hip_flex_r)):
                A.append(np.max(hip_flex_r[ii-30:ii]) - np.min(hip_flex_r[ii-30:ii]))
                B.append(np.max(hip_flex_l[ii-30:ii]) - np.min(hip_flex_l[ii-30:ii])) 
            hip_flex_r = np.asarray(A)
            hip_flex_l = np.asarray(B)
            hip_flexions.append([hip_flex_r, hip_flex_l] if ts else 
                                [np.mean(hip_flex_r), np.mean(hip_flex_l)])
        return hip_flexions

    def _knee_flexions(self, subjects, ts = False):
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
            knee_flexions.append([knee_flex_r, knee_flex_l] if ts else 
                                 [np.mean(knee_flex_r), np.mean(knee_flex_l)])
        return knee_flexions

    def _trunk_rots(self, subjects, ts = False):
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
            # TODO: WHY DO THIS MAX-MIN?
            for ii in range(30,len(trunk_rot_r)):
                A.append(np.max(trunk_rot_r[ii-30:ii]) - np.min(trunk_rot_r[ii-30:ii]))
                B.append(np.max(trunk_rot_l[ii-30:ii]) - np.min(trunk_rot_l[ii-30:ii]))   
            trunk_rot_r = np.asarray(A)
            trunk_rot_l = np.asarray(B)
            trunk_rots.append([trunk_rot_r, trunk_rot_l] if ts else 
                              [np.mean(trunk_rot_r), np.mean(trunk_rot_l)])
        return trunk_rots

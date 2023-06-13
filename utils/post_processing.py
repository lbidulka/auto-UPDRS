import numpy as np
from scipy.signal import savgol_filter, find_peaks
from utils import info, cam_sys_info, alphapose_filtering
from data.body.body_dataset import body_ts_loader
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import copy


def _moving_average(a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        # return np.concatenate([a[:n-1], ret[n - 1:] / n])
        return ret[n - 1:] / n

def get_TUG_time_to_complete(S_id, action_class, in_frames):
    '''
    Computes the time to complete the TUG test and the length of the transition periods using the 
    action classification timeseries.
    '''
    start, walk_start, walk_end, end = None, None, None, None
    # iterate through action class, find transitions
    for i in range(1, len(action_class)):
        prev = action_class[i-1]
        curr = action_class[i]
        # sit -> transition
        if prev == -1 and curr == 0:
            start = in_frames[i]
        # transition -> walki
        elif prev == 0 and curr == 1:
            walk_start = in_frames[i]
        # walk -> transition
        elif prev == 1 and curr == 0:
            walk_end = in_frames[i]
        # transition -> sit
        elif (prev == 0 and curr == -1):
            end = in_frames[i]

    # TODO: BE SURE THIS WORKS FOR ALL
    # if there was no final transition to sitting, then the end is the last frame
    if end is None:
        end = in_frames[-1]

    print("\nstart: ", start, ", walk_start: ", walk_start, ", walk_end: ", walk_end, ", end: ", end)

    fps = cam_sys_info.new_sys_fps if S_id in info.subjects_new_sys else cam_sys_info.old_sys_fps

    time_to_complete = round((end - start) / fps, 2)
    rise_time = round((walk_start - start) / fps, 2)
    sit_time = round((end - walk_end) / fps, 2)
    
    return time_to_complete, rise_time, sit_time

def get_T_thigh_angle(pred_3d, n=0):
    '''
    Computes the angle between the T Neck->Hip vector and the avg of the LL and RL Hip->LKnee vectors
    '''
    # get LL and RL Hip->LKnee vectors
    LL_hip_knee = pred_3d[:, :, info.PD_3D_skeleton_kpt_idxs['LKnee']] - pred_3d[:, :, info.PD_3D_skeleton_kpt_idxs['Hip']]
    RL_hip_knee = pred_3d[:, :, info.PD_3D_skeleton_kpt_idxs['RKnee']] - pred_3d[:, :, info.PD_3D_skeleton_kpt_idxs['Hip']]

    # get T Neck->Hip vector
    T_neck_hip = pred_3d[:, :, info.PD_3D_skeleton_kpt_idxs['Neck']] - pred_3d[:, :, info.PD_3D_skeleton_kpt_idxs['Hip']]

    # avg LL and RL Hip->LKnee vectors, and get angle between that and T Neck->Hip vector
    LL_RL_hip_knee = (LL_hip_knee + RL_hip_knee) / 2
    LL_RL_hip_knee = LL_RL_hip_knee / np.linalg.norm(LL_RL_hip_knee, axis=1, keepdims=True)
    T_neck_hip = T_neck_hip / np.linalg.norm(T_neck_hip, axis=1, keepdims=True)
    T_to_thigh_angle = np.arccos(np.sum(LL_RL_hip_knee * T_neck_hip, axis=1))

    if n != 0:
        return _moving_average(T_to_thigh_angle, n=n)

    return T_to_thigh_angle

def get_d_T_thigh_angle(ang):
    '''
    compute discrete derivative of T_to_thigh_angle
    '''
    ang_diff = np.diff(ang)
    return np.concatenate([[0], ang_diff])
    # return ang_diff

def classify_tug(T_to_thigh_angle, n=5, diff_thresh=0.015, avg_method=False):
    # diff_thresh=0.025
    '''
    Classifies each of the 3D timeseries frames as 
        -1: 'sitting'
         0: 'transitioning'
         1: 'walking'
    using the Torso-to-thigh angle.

    NB: Assumes the subject always starts in a sitting position
    '''
    angle = copy.deepcopy(T_to_thigh_angle)

    if avg_method:
        # simpler, smooth the signal, then threshold
        thresh = 2.5
        avg_n = 5
        transition_abs = np.convolve(angle, np.ones(avg_n) / avg_n, 'same')
        for i in range(10):
            transition_abs = np.convolve(transition_abs, np.ones(avg_n) / avg_n, 'same')
    else:
        for i in range(10):
            angle = np.convolve(angle, np.ones(n) / n, 'same')
            angle[:n] = angle[n:2*n]
            angle[-n:] = angle[-2*n:-n]
        T_to_thigh_angle_diff = get_d_T_thigh_angle(angle)
        # Threshold the absolute d_ang to get transition sections, conv smooths it out
        transition_n = np.int(n*1)
        transition_abs = np.convolve(np.abs(T_to_thigh_angle_diff), np.ones(transition_n) / transition_n, 'same')
        for i in range(3):
            transition_abs = np.convolve(np.abs(transition_abs), np.ones(transition_n) / transition_n, 'same')
        
    transition = transition_abs > diff_thresh

    # initially in state -1 (sitting), in transition switch to state 0 (transitioning), then to invert state after transition
    action = np.zeros(len(angle))
    state = -1
    action[0] = state
    for i in range(1, len(angle)-1):
        prev = transition[i-1]
        curr = transition[i]
        next = transition[i+1]
        # transition is always 0
        if transition[i]:
                action[i] = 0
        # transition is behind, but not now or in front, we have changed from sitting or standing
        elif prev and not curr and not next:
            state *= -1
            action[i] = state
        else:
            action[i] = state
        
        # TODO: WHAT TO DO FOR FINAL POINT? ALWAYS ASSUME RETURN TO SITTING?
        action[-1] = -1

    # return np.concatenate((np.ones(n)*-1, action.astype(int))) , np.concatenate((np.zeros(n), transition_abs))
    return  action.astype(int), transition_abs


class gait_processor():
    '''
    For processing extracted 3D body keypoints 
    '''
    def __init__(self, body_ts_loader, fig_outpath, bone_norm=True, win_feats=True) -> None:
        self.fig_outpath = fig_outpath
        self.body_ts_loader = body_ts_loader
        self.task = body_ts_loader.task
        self.subjects = body_ts_loader.subjects
        self.feat_names = info.clinical_gait_feat_names
        # TODO: just compute ts always, and then avg the ts data to get the avg data
        self.bone_norm = bone_norm  # norm features by hip bone length?
        self.win_feats = win_feats  # use windows for features that might use it? (like Mohsens original)
        self.feats_avg = self.compute_features(self.subjects, ts=False)
        self.feats_ts = self.compute_features(self.subjects, ts=True)
        self.thresholds = None

        self.data_frame_idxs = self.body_ts_loader.data_idxs
         
    
    feat_plot_ylims = {
        'windowed': {
            'Step Width': [0, 1.25],
            'Cadence': [0, 400],
            'Right Arm Swing': [1, 2.5],
            'Left Arm Swing': [1, 2.5],
            
            'Right Step Length': [0, 2],
            'Left Step Length': [0, 2],
            'Right Foot Clearance': [0, 0.75],
            'Left Foot Clearance': [0, 0.75],
            
            'Right Hip Flexion': [0, 0.8],
            'Left Hip Flexion': [0, 0.8],
            'Right Knee Flexion': [0, 0.8],
            'Left Knee Flexion': [0, 0.8],

            'Right Trunk rotation (calculated by right side key points)': [0, 0.75],
            'Left Trunk rotation (calculated by left side key points)': [0, 0.75],
            'Arm swing symmetry': [0.5, 1.5],
        },
        'unwindowed': {
            'Step Width': [0, 1.5],
            'Cadence': [0, 400],
            'Right Arm Swing': [0.5, 2.5],
            'Left Arm Swing': [0.5, 2.5],
            
            'Right Step Length': [0, 2],
            'Left Step Length': [0, 2],
            'Right Foot Clearance': [-0.5, -2.5],
            'Left Foot Clearance': [-0.5, -2.5],
            
            'Right Hip Flexion': [-1, 1],
            'Left Hip Flexion': [-1, 1],
            'Right Knee Flexion': [0, 1.25],
            'Left Knee Flexion': [0, 1.25],

            # 'Right Trunk rotation (calculated by right side key points)': [0, 1.5],
            # 'Left Trunk rotation (calculated by left side key points)': [0, 1.5],
            'Right Trunk rotation (calculated by right side key points)': [-0.5, 0.5],
            'Left Trunk rotation (calculated by left side key points)': [-0.5, 0.5],
            'Arm swing symmetry': [0, 1.75],
        },        
        # Not Procrustes:
        # 'unwindowed': {
        #     'Step Width': [0, 1.25],
        #     'Cadence': [0, 400],
        #     'Right Arm Swing': [0, 3],
        #     'Left Arm Swing': [0, 3],
            
        #     'Right Step Length': [-1, 2],
        #     'Left Step Length': [-1, 2],
        #     'Right Foot Clearance': [-3, 0],
        #     'Left Foot Clearance': [-3, 0],
            
        #     'Right Hip Flexion': [-1, 1],
        #     'Left Hip Flexion': [-1, 1],
        #     'Right Knee Flexion': [1, 2.5],
        #     'Left Knee Flexion': [1, 2.5],

        #     'Right Trunk rotation (calculated by right side key points)': [-6, -1],
        #     'Left Trunk rotation (calculated by left side key points)': [-6, -1],
        #     'Arm swing symmetry': [0, 2],
        # },        

        # 'Right Foot Clearance': [1, 2.5],
        # 'Left Foot Clearance': [1, 2.5],
        # 'Right Step Length': [0, 2.0],
        # 'Left Step Length': [0, 2.0],

        # 'Right Hip Flexion': [-1, 1],
        # 'Left Hip Flexion': [-1, 1],
        # 'Right Knee Flexion': [1, 3],
        # 'Left Knee Flexion': [1, 3],

        # 'Right Trunk rotation (calculated by right side key points)': [-3, 0],
        # 'Left Trunk rotation (calculated by left side key points)': [-2, 2],
        # 'Arm swing symmetry': [0, 2.0],
    }

    # feat_plot_xlims = {
    #     'tug_stand_walk_sit': 100,
    #     'free_form_oval': 2000,
    # }
    
    def plot_feats_ts(self, show=True, save_fig=True, outpath=None, plot_subjs=None, time_ax=False):
        '''
        Plot the time series of the features
    
        Notes:
        - Some subjects have longer time series than others
        - Cadence is not properly the timeseries yet, it is based on peaks
        TODO: CREATE PROPER CADENCE TIMESERIES    
        TODO: use frame idx as x coord, not just pose idx
        '''
        if outpath is None:
            outpath = self.fig_outpath + 'feats_ts.png'
        if plot_subjs is None:
            plot_subjs = self.subjects

        fig_rows = 5
        fig_cols = 3
        fig, ax = plt.subplots(fig_rows, fig_cols, layout="constrained")
        fig.set_size_inches(18.5, 10.5)
        colours = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']

        for i, feat_name in enumerate(info.clinical_gait_feat_names):
            # set lims
            win_token = 'windowed' if self.win_feats else 'unwindowed'
            if feat_name in self.feat_plot_ylims[win_token].keys():
                ax[i//fig_cols, i%fig_cols].set_ylim(self.feat_plot_ylims[win_token][feat_name])

            if self.task == 'tug_stand_walk_sit':
                xlim = 100
            elif self.task == 'free_form_oval':
                xlim = 1850

            # if time_ax:
            #     fps = cam_sys_info.new_sys_fps if S_id in info.subjects_new_sys else cam_sys_info.old_sys_fps
            #     xlim /= fps

            for S_id in plot_subjs:
                j  = self.subjects.index(S_id)
                feat = self.feats_ts[i][j]
                label = S_id if i==0 else None
                frame_idxs = copy.deepcopy(self.data_frame_idxs[j][0])

                if time_ax:
                    fps = cam_sys_info.new_sys_fps if S_id in info.subjects_new_sys else cam_sys_info.old_sys_fps
                    frame_idxs = np.round(frame_idxs / fps, 3)
                    xlim /= fps
                    x_offset = alphapose_filtering.task_timestamps[S_id][self.task]['start']
                    frame_idxs += x_offset
                    xlabel = 'time (s)'
                else:
                    x_offset = 0
                    xlabel = 'frame idx'
                
                # some features reduce length by 30 due to windowing
                # TODO: HOW TO BEST HANDLE CADENCE???
                if feat.shape[0] < frame_idxs.shape[0]:
                    frame_idxs = frame_idxs[:feat.shape[0]]
                
                # get frame_idxs gaps & their idxs, then split there
                gaps = np.diff(frame_idxs) > 1
                gap_idxs = np.where(gaps)[0]
                feat_splits = np.split(feat, gap_idxs+1)    # y
                frame_idxs_splits = np.split(frame_idxs, gap_idxs+1)    # x
                # plot lines
                lines = [list(zip(x, y)) for x, y in zip(frame_idxs_splits, feat_splits)]
                lc = LineCollection(lines, linewidths=1.5, alpha=0.5, label=label, colors=colours[j])
                ax[i//fig_cols, i%fig_cols].add_collection(lc)

            ax[i//fig_cols, i%fig_cols].set_title(info.clinical_gait_feat_names[i])
            ax[i//fig_cols, i%fig_cols].set_xlabel(xlabel)
            ax[i//fig_cols, i%fig_cols].set_xlim([x_offset, xlim])

        fig.legend(loc='right', bbox_to_anchor=(1, 0.5))
        if save_fig: plt.savefig(outpath, dpi=500)
        if show: plt.show()

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
    def set_thresholds(self, subjs=info.healthy_controls, filter='moving_avg'):
        control_feats = self.compute_features(subjs, ts=False, filter=filter)
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

        self.thresholds = np.vstack([thresholds_min, thresholds_max])
        return np.vstack([thresholds_min, thresholds_max])
    
    def compute_features(self, subjects, ts=False, filter='moving_avg'):
        if ts:
            step_widths = self._step_width(subjects, ts, filter=filter)
            step_lengths = self._step_lengths(subjects, ts)
            cadences_gaitspeeds_gaitspeedvars = self._cadence_gaitspeed_gaitspeedvar(subjects, ts)
            foot_lifts = self._foot_lifts(subjects, ts, filter=filter)
            arm_swings = self._arm_swings(subjects, ts, filter=filter)
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
        
        step_widths = np.array(self._step_width(subjects, ts, filter=filter))
        step_lengths = np.array(self._step_lengths(subjects, ts))
        cadences_gaitspeeds_gaitspeedvars = np.array(self._cadence_gaitspeed_gaitspeedvar(subjects, ts))
        foot_lifts = np.array(self._foot_lifts(subjects, ts, filter=filter))
        arm_swings = np.array(self._arm_swings(subjects, ts, filter=filter))
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
    
    def _get_proper_subj_data(self, S_id):
        '''
        Fetches the data for the given subject, only returns data which makes sense to compute gait features on
        '''
        data = self.body_ts_loader.get_data_norm(S_id)

        if self.task == 'tug_stand_walk_sit':
            T_to_thigh_angle = get_T_thigh_angle(np.transpose(data, (0,2,1)))
            
            ang_smooth_n = 10
            T_to_thigh_angle = np.convolve(np.abs(T_to_thigh_angle), np.ones(ang_smooth_n) / ang_smooth_n, 'same')
            T_to_thigh_angle[:ang_smooth_n] = T_to_thigh_angle[ang_smooth_n:2*ang_smooth_n]
            T_to_thigh_angle[-ang_smooth_n:] = T_to_thigh_angle[-2*ang_smooth_n:-ang_smooth_n]

            action_class, _ = classify_tug(T_to_thigh_angle, n=15, diff_thresh=0.015)

            # data = data[classify_tug(np.transpose(data, (0,2,1)))]
            data = data[action_class==1]
        elif self.task == 'free_form_oval':
            pass
        else:
            raise Exception("Task {} not yet implemented".format(self.task))
        return data

    # TODO: Helper to do the windowing
    def _window(self, data, win_len=30, win_step=1):
        pass

    # Helper to apply filtering to time series data
    def _filter_1d(self, data, filter=None, win_len=3, ord=3):
        if filter is None:
            return data
        if filter == 'moving_avg':
            return savgol_filter(data, win_len, 1)
        elif filter == 'savgol':
            return savgol_filter(data, win_len, ord)
    
    def _step_width(self, subjects, ts = False, filter="moving_avg"):
        step_widths = []
        for subj in subjects:
            data_norm = self._get_proper_subj_data(subj)
            step_width = np.abs(data_norm[:,info.PD_3D_skeleton_kpt_idxs['LAnkle'],0] - data_norm[:,info.PD_3D_skeleton_kpt_idxs['RAnkle'],0])
            bone_length = np.linalg.norm((data_norm[:,info.PD_3D_skeleton_kpt_idxs['LHip']] - data_norm[:,info.PD_3D_skeleton_kpt_idxs['RHip']]),axis=-1)
            if self.bone_norm:
                step_width /= bone_length
            step_widths.append(self._filter_1d(step_width, filter) if ts else 
                                np.mean(self._filter_1d(step_width, "moving_avg", 29, 1)))
        return step_widths
    
    def _step_lengths(self, subjects, ts=False, filter=True, bone_norm=True):
        max_win = 30
        step_lengths = []
        for subj in subjects:
            data_norm = self._get_proper_subj_data(subj)
            step_length = np.linalg.norm((data_norm[:,info.PD_3D_skeleton_kpt_idxs['LAnkle']] - data_norm[:,info.PD_3D_skeleton_kpt_idxs['RAnkle']]), axis=-1)
            # IF RIGHT FOOT IS HIGHER THAN LEFT FOOT, & VICE VERSA (IE LIFTED)
            row_r = (data_norm[:,info.PD_3D_skeleton_kpt_idxs['RAnkle'],2] - data_norm[:,info.PD_3D_skeleton_kpt_idxs['LAnkle'],2] > 0)
            row_l = (data_norm[:,info.PD_3D_skeleton_kpt_idxs['LAnkle'],2] - data_norm[:,info.PD_3D_skeleton_kpt_idxs['RAnkle'],2] > 0)
            step_length_r = step_length[row_r]
            step_length_l = step_length[row_l]
            #
            bone_length_r = np.linalg.norm((data_norm[:,info.PD_3D_skeleton_kpt_idxs['RKnee']] - data_norm[:,info.PD_3D_skeleton_kpt_idxs['RAnkle']]), axis=-1)[row_r]
            bone_length_l = np.linalg.norm((data_norm[:,info.PD_3D_skeleton_kpt_idxs['LKnee']] - data_norm[:,info.PD_3D_skeleton_kpt_idxs['LAnkle']]), axis=-1)[row_l]
            if self.bone_norm:
                step_length_r /= bone_length_r
                step_length_l /= bone_length_l

            if not ts:
                A=[]
                for ii in range(max_win, len(step_length_r)):
                    A.append(np.max(step_length_r[ii-max_win:ii]))
                step_length_r = np.asarray(A)
                B=[]
                for ii in range(max_win, len(step_length_l)):
                    B.append(np.max(step_length_l[ii-max_win:ii]))
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
                if self.win_feats:
                    # take max of last 30 timesteps, to get peak step lengths
                    sl_R_peaks = np.full_like(sl_R, 0)
                    sl_L_peaks = np.full_like(sl_L, 0)
                    for ii in range(max_win, len(sl_R)):
                        sl_R_peaks[ii] = np.max(sl_R[ii-max_win:ii])
                    for ii in range(max_win, len(sl_L)):
                        sl_L_peaks[ii] = np.max(sl_L[ii-max_win:ii])
                    # step_lengths.append(np.vstack([sl_R_peaks, sl_L_peaks]))
                step_lengths.append(np.vstack([sl_R, sl_L]))
        return step_lengths
            
    def _cadence_gaitspeed_gaitspeedvar(self, subjects, ts = False):
        cadences_and_speeds = []
        for subj in subjects:
            data_norm = self._get_proper_subj_data(subj)
            toe_traj = np.abs(data_norm[:,info.PD_3D_skeleton_kpt_idxs['RAnkle'],2] - data_norm[:,info.PD_3D_skeleton_kpt_idxs['LAnkle'],2])
            peaks, _ = find_peaks(toe_traj, distance=5, height=-0.2)
            ave = np.mean(toe_traj[peaks]) - 0.3
            peaks, _ = find_peaks(toe_traj, distance=5, height=ave)

            if subj in info.subjects_new_sys:
                cadence = 60 / ((peaks[1:] - peaks[:-1]) / 30)
                gait_speed = toe_traj[peaks[1:]] * cadence
            else:
                cadence = 60 / ((peaks[1:] - peaks[:-1]) / 15)
                gait_speed = toe_traj[peaks[1:]] * cadence
            # TODO: MAP THESE VALUES TO THE TIMESTEPS OF THE DATA
            cadences_and_speeds.append([cadence, gait_speed] if ts else [np.mean(cadence), np.mean(gait_speed)])
        return cadences_and_speeds
    
    def _foot_lifts(self, subjects, ts = False, filter="moving_avg", bone_norm=True):
        max_win = 30
        foot_lifts = []
        for subj in subjects:
            data_norm = self._get_proper_subj_data(subj)
            foot_height_r = data_norm[:,info.PD_3D_skeleton_kpt_idxs['RAnkle'],1] - data_norm[:,info.PD_3D_skeleton_kpt_idxs['RHip'],1]
            foot_height_l = data_norm[:,info.PD_3D_skeleton_kpt_idxs['LAnkle'],1] - data_norm[:,info.PD_3D_skeleton_kpt_idxs['LHip'],1]
            bone_length_l = np.linalg.norm((data_norm[:,info.PD_3D_skeleton_kpt_idxs['LHip']] - data_norm[:,info.PD_3D_skeleton_kpt_idxs['LKnee']]), axis=-1)
            bone_length_r = np.linalg.norm((data_norm[:,info.PD_3D_skeleton_kpt_idxs['RKnee']] - data_norm[:,info.PD_3D_skeleton_kpt_idxs['RHip']]), axis=-1)
            if self.bone_norm:
                foot_height_r /= bone_length_r
                foot_height_l /= bone_length_l
            # TODO: REPLACE WITH INTERNAL WINDOWING FUNC
            if self.win_feats:
                A=[]
                B=[]
                for ii in range(max_win, len(foot_height_r)):
                    A.append(np.max(foot_height_r[ii-max_win:ii]) - np.min(foot_height_r[ii-max_win:ii]))
                    B.append(np.max(foot_height_l[ii-max_win:ii]) - np.min(foot_height_l[ii-max_win:ii]))
                foot_height_n_r = np.asarray(A)
                foot_height_n_l = np.asarray(B)
            else:
                foot_height_n_r = foot_height_r
                foot_height_n_l = foot_height_l
            foot_lifts.append([self._filter_1d(foot_height_n_r, filter), self._filter_1d(foot_height_n_l, filter)] if ts else 
                              [np.mean(foot_height_n_r), np.mean(foot_height_n_l)])   
        return foot_lifts

    def _arm_swings(self, subjects, ts = False, filter="moving_avg", bone_norm=True):
        arm_swings = []
        for subj in subjects:
            data_norm = self._get_proper_subj_data(subj)
            bone_length = np.linalg.norm((data_norm[:,info.PD_3D_skeleton_kpt_idxs['RHip']] - data_norm[:,info.PD_3D_skeleton_kpt_idxs['LHip']]), axis=-1)
            dist_R = np.linalg.norm((data_norm[:,info.PD_3D_skeleton_kpt_idxs['LWrist']] - data_norm[:,info.PD_3D_skeleton_kpt_idxs['RHip']]), axis=-1)
            dist_L = np.linalg.norm((data_norm[:,info.PD_3D_skeleton_kpt_idxs['RWrist']] - data_norm[:,info.PD_3D_skeleton_kpt_idxs['LHip']]), axis=-1)
            if self.bone_norm:
                dist_R /= bone_length
                dist_L /= bone_length
            dist_R = self._filter_1d(dist_R, filter)
            dist_L = self._filter_1d(dist_L, filter)
            arm_swings.append([dist_R, dist_L, dist_L / dist_R] if ts else
                                [np.mean(dist_R), np.mean(dist_L), np.mean(dist_L) / np.mean(dist_R)])
        return arm_swings
    
    def _hip_flexions(self, subjects, ts = False, bone_norm=True):
        max_win = 30
        hip_flexions = []
        for subj in subjects:
            data_norm = self._get_proper_subj_data(subj)
            dist_l = data_norm[:,info.PD_3D_skeleton_kpt_idxs['LHip'],2] - data_norm[:,info.PD_3D_skeleton_kpt_idxs['LKnee'],2] # y dist
            dist_r = data_norm[:,info.PD_3D_skeleton_kpt_idxs['RHip'],2] - data_norm[:,info.PD_3D_skeleton_kpt_idxs['RKnee'],2] # y dist
            bone_l = np.linalg.norm((data_norm[:,info.PD_3D_skeleton_kpt_idxs['LHip']] - data_norm[:,info.PD_3D_skeleton_kpt_idxs['LKnee']]), axis=-1)
            bone_r = np.linalg.norm((data_norm[:,info.PD_3D_skeleton_kpt_idxs['RHip']] - data_norm[:,info.PD_3D_skeleton_kpt_idxs['RKnee']]), axis=-1)
            if self.bone_norm:
                hip_flex_r = dist_r / bone_r
                hip_flex_l = dist_l / bone_l
            else:
                hip_flex_r = dist_r
                hip_flex_l = dist_l
            if self.win_feats:
                A = []
                B = []
                for ii in range(max_win, len(hip_flex_r)):
                    A.append(np.max(hip_flex_r[ii-max_win:ii]) - np.min(hip_flex_r[ii-max_win:ii]))
                    B.append(np.max(hip_flex_l[ii-max_win:ii]) - np.min(hip_flex_l[ii-max_win:ii])) 
                hip_flex_r = np.asarray(A)
                hip_flex_l = np.asarray(B)
            hip_flexions.append([hip_flex_r, hip_flex_l] if ts else 
                                [np.mean(hip_flex_r), np.mean(hip_flex_l)])
        return hip_flexions

    def _knee_flexions(self, subjects, ts = False):
        max_win = 30
        knee_flexions = []
        for subj in subjects:
            data_norm = self._get_proper_subj_data(subj)
            thigh_l = np.linalg.norm((data_norm[:,info.PD_3D_skeleton_kpt_idxs['LHip']] - data_norm[:,info.PD_3D_skeleton_kpt_idxs['LKnee']]), axis=-1)
            thigh_r = np.linalg.norm((data_norm[:,info.PD_3D_skeleton_kpt_idxs['RHip']] - data_norm[:,info.PD_3D_skeleton_kpt_idxs['RKnee']]), axis=-1)
            shin_l = np.linalg.norm((data_norm[:,info.PD_3D_skeleton_kpt_idxs['LKnee']] - data_norm[:,info.PD_3D_skeleton_kpt_idxs['LAnkle']]), axis=-1)
            shin_r = np.linalg.norm((data_norm[:,info.PD_3D_skeleton_kpt_idxs['RKnee']] - data_norm[:,info.PD_3D_skeleton_kpt_idxs['RAnkle']]), axis=-1)
            leg_l = np.linalg.norm((data_norm[:,info.PD_3D_skeleton_kpt_idxs['LHip']] - data_norm[:,info.PD_3D_skeleton_kpt_idxs['LAnkle']]), axis=-1)
            leg_r = np.linalg.norm((data_norm[:,info.PD_3D_skeleton_kpt_idxs['RHip']] - data_norm[:,info.PD_3D_skeleton_kpt_idxs['RAnkle']]), axis=-1)
            knee_flex_r = ((leg_r**2 / (thigh_r * shin_r)) - (thigh_r / shin_r) - (shin_r / thigh_r)) / (2)
            knee_flex_l = ((leg_l**2 / (thigh_l * shin_l)) - (thigh_l / shin_l) - (shin_l / thigh_l)) / (2)
            if self.win_feats:
                A = []
                B = []
                for ii in range(max_win,len(knee_flex_r)):
                    A.append(np.max(knee_flex_r[ii-max_win:ii]) - np.min(knee_flex_r[ii-max_win:ii]))
                    B.append(np.max(knee_flex_l[ii-max_win:ii]) - np.min(knee_flex_l[ii-max_win:ii]))
                knee_flex_r = np.asarray(A)
                knee_flex_l = np.asarray(B)
            # TODO: IS THIS THE RIGHT ORDER? MOHSEN HAD IT SWAPPED (L, R), OPPOSITE TO ALL OTHERS
            knee_flexions.append([knee_flex_r, knee_flex_l] if ts else 
                                 [np.mean(knee_flex_r), np.mean(knee_flex_l)])
        return knee_flexions

    def _trunk_rots(self, subjects, ts=False):
        max_win = 30
        trunk_rots = []
        for subj in subjects:
            data_norm = self._get_proper_subj_data(subj)
            data_normal = data_norm - data_norm[:,:info.PD_3D_skeleton_kpt_idxs['LHip']]
            shoulder_l = np.linalg.norm((data_normal[:,info.PD_3D_skeleton_kpt_idxs['LShoulder'],[0,1]] - data_normal[:,info.PD_3D_skeleton_kpt_idxs['LHip'],[0,1]]), axis=-1)
            shoulder_r = np.linalg.norm((data_normal[:,info.PD_3D_skeleton_kpt_idxs['RShoulder'],[0,1]] - data_normal[:,info.PD_3D_skeleton_kpt_idxs['RHip'],[0,1]]), axis=-1)
            hip_l = np.linalg.norm((data_normal[:,info.PD_3D_skeleton_kpt_idxs['Hip'],[0,1]] - data_normal[:,info.PD_3D_skeleton_kpt_idxs['LHip'],[0,1]]), axis=-1)
            hip_r = np.linalg.norm((data_normal[:,info.PD_3D_skeleton_kpt_idxs['Hip'],[0,1]] - data_normal[:,info.PD_3D_skeleton_kpt_idxs['RHip'],[0,1]]), axis=-1)
            hip2shoulder_l = np.linalg.norm((data_normal[:,info.PD_3D_skeleton_kpt_idxs['RHip'],[0,1]] - data_normal[:,info.PD_3D_skeleton_kpt_idxs['RShoulder'],[0,1]]), axis=-1)
            hip2shoulder_r = np.linalg.norm((data_normal[:,info.PD_3D_skeleton_kpt_idxs['LHip'],[0,1]] - data_normal[:,info.PD_3D_skeleton_kpt_idxs['LShoulder'],[0,1]]), axis=-1)
            trunk_rot_r = ((hip2shoulder_r**2 / (hip_r * shoulder_r)) - (hip_r / shoulder_r) - (shoulder_r / hip_r)) / (2)  # TODO: WHY IS THE SIGN FLIPPED?
            trunk_rot_l = ((hip2shoulder_l**2 / (hip_l * shoulder_l)) - (hip_l / shoulder_l) - (shoulder_l / hip_l)) / (-2)
            if self.win_feats:
                A = []
                B = []
                for ii in range(max_win, len(trunk_rot_r)):
                    A.append(np.max(trunk_rot_r[ii-max_win:ii]) - np.min(trunk_rot_r[ii-max_win:ii]))
                    B.append(np.max(trunk_rot_l[ii-max_win:ii]) - np.min(trunk_rot_l[ii-max_win:ii]))   
                trunk_rot_r = np.asarray(A)
                trunk_rot_l = np.asarray(B)
            trunk_rots.append([trunk_rot_r, trunk_rot_l] if ts else 
                            [np.mean(trunk_rot_r), np.mean(trunk_rot_l)])
        return trunk_rots

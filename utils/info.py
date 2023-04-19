import numpy as np

# Subject ID information
subjects_PD       = ['S01','S02','S03','S04','S05','S06','S07','S09',
                     'S10','S11','S12','S13','S14','S16','S17','S18',
                     'S19','S21','S22','S23','S24','S28','S29','S30',
                     'S31','S32','S33','S34','S35']

healthy_controls  = ['S08','S20','S25','S26','S27']

subjects_All      = ['S01',       'S02',     'S03',     'S04',     'S05',
                     'S06',       'S07',     'S08',     'S09',     'S10',
                     'S11',       'S12',     'S13',     'S14',     'S15', 
                     'S16',       'S17',     'S18',     'S19',     'S20',
                     'S21',       'S22',     'S23',     'S24',     'S25',
                     'S26',       'S27',     'S28',     'S29',     'S30',
                     'S31',       'S32',     'S33',     'S34',     'S35']

subjects_All_date = ['20210223','20191114','20191120','20191112','20191119',
                     '20200220','20191121','20191126','20191128','20191203',
                     '20191204','20200108','20200109','20200121','20200122',
                     '20200123','20200124','20200127','20200130','20200205',
                     '20200206','20200207','20200213','20200214','20200218',
                     '20191126','20200221','20210706','20210804','20200206',
                     '20210811','20191210','20191212','20191218','20200227']

# These subjs were capture with the new system, and are used for training since 
# they have synchronized cams, no frame drops, and higher frame rate of 30fps
subjects_new_sys = ['S01', 'S28', 'S29', 'S31']
new_sys_vid_suffixes = {
    'S01': {'free_form_oval': '20210223144019_20210223144243',
            'tug_stand_walk_sit': '20210223145241_20210223145331'},
    'S28': {'free_form_oval': '20210706134648_20210706134920',
            'tug_stand_walk_sit': '20210706135744_20210706135801'},
    'S29': {'free_form_oval': '20210804172455_20210804172705',
            'tug_stand_walk_sit': '20210804173404_20210804173419'},
    'S31': {'free_form_oval': '20210811135008_20210811135233',
            'tug_stand_walk_sit': '20210811140141_20210811140219'},
}

# Subject ID mapping
subjects_ALL_id_dict = {
                'S01': 9291, 'S02': 9739, 'S03': 9285, 'S04': 9769, 'S05': 9964, 
                'S06': 9746, 'S07': 9270, 'S08': 7399, 'S09': 9283, 'S10': 9107, 
                'S11': 9455, 'S12': 9713, 'S13': 9317, 'S14': 9210, 'S15': 9403,
                'S16': 9791, 'S17': 9813, 'S18': 9525, 'S19': 9419, 'S20': 7532, 
                'S21': 9339, 'S22': 9754, 'S23': 9392, 'S24': 9810, 'S25': 7339, 
                'S26': 7399, 'S27': 7182, 'S28': 9986, 'S29': 9731, 'S30': 9629, 
                'S31': 9314, 'S32': 9448, 'S33': 9993, 'S34': 9182, 'S35': 9351,
                }   

# The big man. We can build him... we have the technology...
subjects_big_dict = {
                'S01': {'id': 9291, 'date': '20210223', 'path': None}, 'S02': {'id': 9739, 'date': '20191114', 'path': None}, 
                'S03': {'id': 9285, 'date': '20191120', 'path': None}, 'S04': {'id': 9769, 'date': '20191112', 'path': None}, 
                'S05': {'id': 9964, 'date': '20191119', 'path': None}, 'S06': {'id': 9746, 'date': '20200220', 'path': None}, 
                'S07': {'id': 9270, 'date': '20191121', 'path': None}, 'S08': {'id': 7399, 'date': '20191126', 'path': None}, 
                'S09': {'id': 9283, 'date': '20191128', 'path': None}, 'S10': {'id': 9107, 'date': '20191203', 'path': None}, 
                'S11': {'id': 9455, 'date': '20191204', 'path': None}, 'S12': {'id': 9713, 'date': '20200108', 'path': None}, 
                'S13': {'id': 9317, 'date': '20200109', 'path': None}, 'S14': {'id': 9210, 'date': '20200121', 'path': None}, 
                'S15': {'id': 9403, 'date': '20200122', 'path': None}, 'S16': {'id': 9791, 'date': '20200123', 'path': None}, 
                'S17': {'id': 9813, 'date': '20200124', 'path': None}, 'S18': {'id': 9525, 'date': '20200127', 'path': None}, 
                'S19': {'id': 9419, 'date': '20200130', 'path': None}, 'S20': {'id': 7532, 'date': '20200205', 'path': None}, 
                'S21': {'id': 9339, 'date': '20200206', 'path': None}, 'S22': {'id': 9754, 'date': '20200207', 'path': None}, 
                'S23': {'id': 9392, 'date': '20200213', 'path': None}, 'S24': {'id': 9810, 'date': '20200214', 'path': None}, 
                'S25': {'id': 7339, 'date': '20200218', 'path': None}, 'S26': {'id': 7399, 'date': '20191126', 'path': None}, 
                'S27': {'id': 7182, 'date': '20200221', 'path': None}, 'S28': {'id': 9986, 'date': '20210706', 'path': None}, 
                'S29': {'id': 9731, 'date': '20210804', 'path': None}, 'S30': {'id': 9629, 'date': '20200206', 'path': None}, 
                'S31': {'id': 9314, 'date': '20210811', 'path': None}, 'S32': {'id': 9448, 'date': '20191210', 'path': None}, 
                'S33': {'id': 9993, 'date': '20191212', 'path': None}, 'S34': {'id': 9182, 'date': '20191218', 'path': None}, 
                'S35': {'id': 9351, 'date': '20200227', 'path': None},
                }   

subjects_ALL_path_dict = {}
for i, subj in enumerate(subjects_All):
    subjects_ALL_path_dict[subj] = subjects_All_date[i] + '/' + str(subjects_ALL_id_dict[subj]) + '/'

# Clinically-informed gait feature names
clinical_gait_feat_names = ['Step Width','Right Step Length', 'Left Step Length', 'Cadence','Right Foot Clearance', 
                            'Left Foot Clearance', 'Right Arm Swing', 'Left Arm Swing', 'Right Hip Flexion', 
                            'Left Hip Flexion', 'Right Knee Flexion','Left Knee Flexion',
                            'Right Trunk rotation (calculated by right side key points)',
                            'Left Trunk rotation (calculated by left side key points)', 'Arm swing symmetry']
clinical_gait_feat_acronyms = ['SW','RSL', 'LSL', 'Cad','RFC', 'LFC', 'RHP', 'LHP', 
                               'RHF', 'LHF', 'RKF','LKF', 'RTR', 'LTR', 'ASS']
# Groups: [Hand, Step, Foot, Hip, Knee, Trunk, Cadence, Arm Swing Symmetry]
clinical_gait_feat_acronyms_group = [['RHP', 'LHP', 'ASS'], ['RSL', 'LSL'], ['RFC', 'LFC'], 
                                     ['RHF', 'LHF'], ['RKF', 'LKF'], ['RTR', 'LTR'], 'Cad', 'SW']

# Labels for the PD subject UPDRS scores (1 if UPDRS score > 0)
Y_true = np.asarray([1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1])


# Definition of pose kpt skeleton (mainly for plotting)
PD_2D_AP_skeleton = {
        'LA': [7, 12, 13, 14],  # L Arm:  Neck, LShoulder, LElbow, LWrist
        'RA': [7, 9, 10, 11],   # R Arm:  Neck, RShoulder, RElbow, RWrist
        'LL': [0, 1, 2, 3],     # L Leg:  Hip, LHip, LKnee, LAnkle
        'RL': [0, 4, 5, 6],     # R Leg:  Hip, RHip, RKnee, RAnkle
        'T': [8, 7, 0],         # Torso:  Head, Neck, Hip
}
PD_3D_lifter_skeleton = {
        'LA': [7, 12, 13, 14], # L Arm:  Neck, LShoulder, LElbow, LWrist
        'RA': [7, 9, 10, 11],  # R Arm:  Neck, RShoulder, RElbow, RWrist
        'LL': [0, 1, 2, 3],    # L Leg:  Hip, LHip, LKnee, LAnkle
        'RL': [0, 4, 5, 6],    # R Leg:  Hip, RHip, RKnee, RAnkle
        'T': [8, 7, 0],        # Torso:  Head, Neck, Hip
}
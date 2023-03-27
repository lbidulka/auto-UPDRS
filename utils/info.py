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


# Subject ID mapping
subjects_ALL_id_dict = {
                'S01': 9291, 'S02': 9739, 'S03': 9285, 'S04': 9769, 'S05': 9964, 
                'S06': 9746, 'S07': 9270, 'S08': 7399, 'S09': 9283, 'S10': 9107, 
                'S11': 9455, 'S12': 9713, 'S13': 9317, 'S14': 9210, 'S15': 9403,
                'S16': 9791, 'S17': 9813, 'S18': 9525, 'S19': 9419, 'S20': 7532, 
                'S21': 7532, 'S22': 9339, 'S23': 9392, 'S24': 9810, 'S25': 7339, 
                'S26': 7399, 'S27': 7182, 'S28': 9986, 'S29': 9731, 'S30': 9629, 
                'S31': 9314, 'S32': 9448, 'S33': 9993, 'S34': 9182, 'S35': 9351,
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
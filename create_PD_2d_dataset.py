import argparse
import os
import fnmatch
from types import SimpleNamespace

# Subject ID mapping -- I HATE THIS BUT I DONT WANNA DEAL WITH MODULE IMPORTING
subjects_ALL_id_dict = {
            'S01': 9291, 'S02': 9739, 'S03': 9285, 'S04': 9769, 'S05': 9964, 
            'S06': 9746, 'S07': 9270, 'S08': 7399, 'S09': 9283, 'S10': 9107, 
            'S11': 9455, 'S12': 9713, 'S13': 9317, 'S14': 9210, 'S15': 9403,
            'S16': 9791, 'S17': 9813, 'S18': 9525, 'S19': 9419, 'S20': 7532, 
            'S21': 7532, 'S22': 9339, 'S23': 9392, 'S24': 9810, 'S25': 7339, 
            'S26': 7399, 'S27': 7182, 'S28': 9986, 'S29': 9731, 'S30': 9629, 
            'S31': 9314, 'S32': 9448, 'S33': 9993, 'S34': 9182, 'S35': 9351,
            }     
subjects_All_date = ['20210223','20191114','20191120','20191112','20191119',
                    '20200220','20191121','20191126','20191128','20191203',
                    '20191204','20200108','20200109','20200121','20200122',
                    '20200123','20200124','20200127','20200130','20200205',
                    '20200206','20200207','20200213','20200214','20200218',
                    '20191126','20200221','20210706','20210804','20200206',
                    '20210811','20191210','20191212','20191218','20200227']
subjects_All  = ['S01',       'S02',     'S03',     'S04',     'S05',
                    'S06',       'S07',     'S08',     'S09',     'S10',
                    'S11',       'S12',     'S13',     'S14',     'S15', 
                    'S16',       'S17',     'S18',     'S19',     'S20',
                    'S21',       'S22',     'S23',     'S24',     'S25',
                    'S26',       'S27',     'S28',     'S29',     'S30',
                    'S31',       'S32',     'S33',     'S34',     'S35']

subjects_new_system = ['S01', 'S28', 'S29', 'S31']
new_sys_vid_suffixes = {
    'S01': {'free_form_oval': '20210223144019_20210223144243',
            'tug_stand_walk_sit': '20210223145241_20210223145331'},
    'S28': {'free_form_oval': '20210706134648_20210706134920',
            'tug_stand_walk_sit': '20210706135743_20210706135801'},
    'S29': {'free_form_oval': '20210804172455_20210804172705',
            'tug_stand_walk_sit': '20210804173404_20210804173419'},
    'S31': {'free_form_oval': '20210811135008_20210811135233',
            'tug_stand_walk_sit': '20210811140141_20210811140219'},
}

class cd:
    """
    Context manager for changing the current working directory, 
    from:  https://stackoverflow.com/questions/431684/equivalent-of-shell-cd-command-to-change-the-working-directory
    """
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def get_AlphaPoses(config):
    '''
    Gets AlphaPose 2D pose predictions from input CAMERA dataset videos
    '''
    print("\nGetting AlphaPose predictions...")
    for S_id, id in subjects_ALL_id_dict.items():
        if S_id in config.subjs_to_get_preds:
            for ch in config.chs:
                print("\n--- {}: CH_{} ---".format(S_id, ch))
                # Construct paths
                subj_idx = subjects_All.index(S_id)
                # New system subjects
                if S_id in subjects_new_system:
                    file_subpath = '{}/'.format(subjects_All_date[subj_idx])
                    if S_id == 'S01':
                        file_subpath += '{}/{}/CH_{}/'.format(subjects_ALL_id_dict[S_id], 'Video Data', ch)
                    elif S_id == 'S28':
                        file_subpath += '{}/'.format(subjects_ALL_id_dict[S_id])
                    # if S_id == 'S29':
                    elif S_id == 'S31':
                        file_subpath += '2021-8-11/'
                    file_subpath += 'LNR616X_ch{}_main_{}.avi'.format(ch[-1], new_sys_vid_suffixes[S_id][config.task])
                # Old system subjects
                if S_id not in subjects_new_system:
                    file_subpath = subjects_All_date[subj_idx] + '/' + str(id)
                    if os.path.exists(config.videos_path + file_subpath + '/Video Data/'):
                        file_subpath += '/Video Data/'
                    elif os.path.exists(config.videos_path + file_subpath + '/Video_Data/'):
                        file_subpath += '/Video_Data/'
                    else:
                        print("ERR not sure what to do here... : ", S_id, id, subjects_All_date[subj_idx])
                        break
                    file_subpath += 'CH_' + ch + '/'
                    for file in os.listdir(config.videos_path + file_subpath):
                        if fnmatch.fnmatch(file, '*' + config.task + '*.mp4'):
                            file_subpath += file

                in_path = config.videos_path + file_subpath
                out_path = config.frames_path + str(id) + '/' + config.task + '/'

                # Make sure alles gut
                print("input video subpath: {} \npreds json outpath: {}".format(file_subpath, out_path))
                if not os.path.exists(out_path):
                    print("No existing out_path dir, creating it as: ", out_path)
                    os.makedirs(out_path)
                if not os.path.exists(in_path):
                    print("ERR Input file not found: ", file_subpath, "\n")

                # Do the thing if alles gut
                else:
                    if config.limit_num_frames:
                        num_frames = str(60 * config.lim_secs) if S_id in subjects_new_system else str(15 * config.lim_secs)
                    with cd(config.AP_dir):
                        # Run AlphaPose on the data
                        ap_cmd = "python3 {} --cfg \"{}\" --checkpoint \"{}\" ".format("scripts/demo_inference.py", 
                                                                              "configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml", 
                                                                              "pretrained_models/halpe26_fast_res50_256x192.pth")
                        ap_cmd += "--sp --debug --gpu 0,1 --detbatch 1 --posebatch 30 --qsize 128 "
                        # ap_cmd += "--debug True --qsize 512 --posebatch 32 "
                        ap_cmd += "--maxframes {} --video \"{}\" --outdir \"{}\"".format(num_frames, in_path, out_path)
                        print("\nTrying AP cmd: ", ap_cmd, end='\n\n')
                        os.system(ap_cmd)
                        
                        # Reformat the output json file
                        mv_cmd = "mv {}alphapose-results.json {}CH{}_alphapose-results.json".format(out_path, out_path, ch)
                        print("Trying mv cmd: ", mv_cmd, end='\n\n')
                        os.system(mv_cmd)
                    print("Done with ", S_id, "CH_" + ch)

def get_config():
    config = SimpleNamespace()
    # Tasks
    config.get_2d_preds = True  # Proces videos to get 2d preds?
    # config.subjs_to_get_preds = subjects_All
    # config.subjs_to_get_preds = subjects_new_system
    config.subjs_to_get_preds = ["S29"]

    # Paths
    config.root_dir = os.path.dirname(os.path.realpath(__file__))
    config.videos_path = "/mnt/CAMERA-data/CAMERA/CAMERA visits/Mobility Visit/Study Subjects/"
    config.frames_path = "/mnt/CAMERA-data/CAMERA/Other/lbidulka_dataset/"
    config.AP_dir = config.root_dir + "/AlphaPose"
    # Settings
    config.limit_num_frames = True  # DEBUG: limit number of frames to process
    config.lim_secs = 5 # Mohsen used 2 min per video
    config.task = "free_form_oval"
    # config.task = "tug_stand_walk_sit"
    config.chs = ["003", "004"]
    return config

def main():
    config = get_config()

    if config.get_2d_preds:
        get_AlphaPoses(config)

if __name__ == '__main__':
    main()

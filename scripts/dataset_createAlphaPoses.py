import cv2
import argparse
import os
import fnmatch
from tqdm import tqdm
import subprocess

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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_path", default="/mnt/CAMERA-data/CAMERA/CAMERA visits/Mobility Visit/Study Subjects/", help="input video dataset path", type=str)
    parser.add_argument("--frames_path", default="/mnt/CAMERA-data/CAMERA/Other/lbidulka_dataset/", help="output frame data path", type=str)
    # parser.add_argument("--out_path", default="./auto_UPDRS/data/2d/", help="output frame data path", type=str)    
    return parser.parse_args()


def main():
    '''
    Gets AlphaPose 2D pose predictions from sequences of png frames
    '''
    input_args = get_args()

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
    

    task = "free_form_oval" # "tug_stand_walk_sit_"
    chs = ["003", "004"]
    num_frames = str(15 * 30) # 15fps * num_sec (15 * (60 * 2) = 2 minutes is what Mohsen used)

    # iterate over the subjects_ALL_id_dict dict of (S_id: id) pairs
    for S_id, id in subjects_ALL_id_dict.items():
        if S_id in ['S02', 'S25', 'S26', 'S27', 'S28', 'S29']: # TEMP FOR TESTING
            for ch in chs:
                print("Trying ", S_id, "CH_" + ch, end='')
                # Construct paths
                subj_idx = subjects_All.index(S_id)
                # in_path_end = str(id) + '/' + task + '/' + ch + '/frames/'
                # in_frames = input_args.frames_path + in_path_end
                in_file_end = subjects_All_date[subj_idx] + '/' + str(id)

                # Deal with bad formatting :/
                if os.path.exists(input_args.videos_path + in_file_end + '/Video Data/'):
                    in_file_end += '/Video Data/'
                elif os.path.exists(input_args.videos_path + in_file_end + '/Video_Data/'):
                    in_file_end += '/Video_Data/'
                else:
                    print("\n  ERR its a really badly formatted one... : ", S_id, id, subjects_All_date[subj_idx])
                    break
                    
                in_file_end += 'CH_' + ch + '/'
                
                for file in os.listdir(input_args.videos_path + in_file_end):
                    if fnmatch.fnmatch(file, '*' + task + '*.mp4'):
                        in_file_end += file
                # Check that the input vid is valid (is mp4)
                if in_file_end[-4:] != '.mp4':
                    print("\n  ERR input file not found (bad name formatting?): ", in_file_end)
                    break

                in_file = input_args.videos_path + in_file_end

                out_path = input_args.frames_path + str(id) + '/' + task + '/'
                
                print(" from: ", in_file_end)

                # Make sure its all good to go
                # if len(os.listdir(in_frames)) == 0:
                #     print("  ERR Input frames folder empty, skipping: ", in_file_end)
                if not os.path.exists(out_path):
                    print("Creating output directory: ", out_path)
                    os.makedirs(out_path)
                if not os.path.exists(in_file):
                    print("  ERR Input file not found: ", in_file_end)
                else:
                    # Run AlphaPose on the frames
                    with cd("./auto_UPDRS/AlphaPose"):
                        os.system("python3 scripts/demo_inference.py \
                                    --cfg configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml \
                                    --checkpoint pretrained_models/halpe26_fast_res50_256x192.pth \
                                    --sp \
                                    --debug \
                                    --gpu 0,1 \
                                    --detbatch 1\
                                    --posebatch 30 \
                                    --qsize 128 \
                                    --video \"" + in_file + "\""
                                    " --outdir \"" + out_path + "\"")
                                    # --debug True \
                                    # --maxframes " + num_frames + 
                                    # --qsize 512 \
                                    # --posebatch 32 \
                                    # --indir " + in_frames +"\
                        # rename the output file
                        os.system("mv " + out_path + "alphapose-results.json " + \
                                out_path + "CH" + ch +"_alphapose-results" + ".json")
                    print("  Success.\n")

if __name__ == '__main__':
    main()

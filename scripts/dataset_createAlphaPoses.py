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
                    'S11': 9455, 'S12': 9713, 'S13': 9317, 'S14': 9210, 'S16': 9403, 
                    'S17': 9791, 'S18': 9813, 'S19': 9525, 'S20': 9419, 'S21': 7532, 
                    'S22': 9339, 'S23': 9754, 'S24': 9392, 'S25': 9810, 'S26': 7339, 
                    'S27': 7399, 'S28': 7182, 'S29': 9986, 'S30': 9731, 'S31': 9629,  
                    'S32': 9314, 'S33': 9448, 'S34': 9993, 'S35': 9182,  
                    }        
    subjects_All_date = ['20210223','20191114','20191120','20191112','20191119','20200220','20191121',
                    '20191126','20191128','20191203','20191204','20200108','20200109','20200121','20200122','20200123',
                    '20200124','20200127','20200130','20200205','20200206','20200207','20200213','20200214','20200218',
                    '20191126','20200221','20210706','20210804','20200206','20210811','20191210','20191212','20191218',
                    '20200227']
    subjects_All = ['S01','S02','S03','S04','S05',
                'S06','S07','S08','S09','S10',
                'S11','S12','S13','S14','S16',
                'S17','S18','S19','S20','S21',
                'S22','S23','S24','S25','S26',
                'S27','S28','S29','S30','S31',
                'S32','S33','S34','S35']
    

    task = "free_form_oval" #"tug_stand_walk_sit_"
    chs = ["003", "004"]


    print("Looking in: ", input_args.frames_path)
    # iterate over the subjects_ALL_id_dict dict of (S_id: id) pairs
    for S_id, id in subjects_ALL_id_dict.items():
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
                                --gpu 0,1 \
                                --video \"" + in_file + "\" \
                                --maxframes " + str(15 * 60 * 2) + " \
                                --outdir \"" + out_path + "\"")
                                # --qsize 512 \
                                # --posebatch 32 \
                                # --indir " + in_frames +"\
                    # rename the output file
                    os.system("mv " + out_path + "alphapose-results.json " + \
                              out_path + "CH" + ch +"_alphapose-results" + ".json")
                print("  Success.")

if __name__ == '__main__':
    main()

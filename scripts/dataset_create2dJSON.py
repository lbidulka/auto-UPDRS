import argparse
import os
import sys
import pickle
 
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from data.body.body_dataset import get_2D_keypoints_from_alphapose_dict

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
    parser.add_argument("--out_path", default="./auto_UPDRS/data/body/2d_proposals/", help="output frame data path", type=str)
    parser.add_argument("--dataset_path", default="/mnt/CAMERA-data/CAMERA/Other/lbidulka_dataset/", help="output frame data path", type=str)
    return parser.parse_args()


def main():
    '''
    Builds a combined task dataset (JSON) of 2D pose predictions from per-subject-per-channel alphapose pred JSONs
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

    # init the dict
    kpts_dict = {}

    # iterate over the subjects_ALL_id_dict dict of (S_id: id) pairs
    for S_id, id in subjects_ALL_id_dict.items():
        for ch in chs:
            print("Trying ", S_id, "CH_" + ch, end='')
            # Construct paths
            in_json_name =  str(id) + '/' + task + '/' + 'CH' + ch + "_alphapose-results.json"
            in_json_pth = input_args.dataset_path + in_json_name

            if not os.path.exists(in_json_pth):
                print("  ERR Input JSON not found: ", in_json_name)
            else:
                kpts_dict = get_2D_keypoints_from_alphapose_dict(in_json_pth, ch, S_id, task, kpts_dict, norm_cam=True)
                print("  Successfully loaded alphapose data.")

    # Pickle the 2d keypoints dict
    out_name = input_args.out_path + task + "_2D_kpts.pickle"
    print("Saving to ", out_name)

    with open(out_name, 'wb') as handle:
        pickle.dump(kpts_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Successfully created 2D keypoint dataset.")


if __name__ == '__main__':
    main()

import os
import fnmatch
import pickle
from types import SimpleNamespace

from utils.info import subjects_ALL_id_dict, subjects_All_date, subjects_All, subjects_new_sys, new_sys_vid_suffixes
from data.body.body_dataset import filter_alphapose_results

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
                if S_id in subjects_new_sys:
                    file_subpath = '{}/'.format(subjects_All_date[subj_idx])
                    if S_id == 'S01':
                        file_subpath += '{}/{}/CH_{}/'.format(subjects_ALL_id_dict[S_id], 'Video Data', ch)
                    elif S_id == 'S28':
                        file_subpath += '{}/'.format(subjects_ALL_id_dict[S_id])
                    # if S_id == 'S29':
                    elif S_id == 'S31':
                        file_subpath += '2021-8-11/'
                    file_subpath += 'LNR616X_ch{}_main_{}.avi'.format(ch[-1], new_sys_vid_suffixes[S_id][config.updrs_task])
                # Old system subjects
                if S_id not in subjects_new_sys:
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
                        if fnmatch.fnmatch(file, '*' + config.updrs_task + '*.mp4'):
                            file_subpath += file

                in_path = config.videos_path + file_subpath
                out_path = config.dataset_path + str(id) + '/' + config.updrs_task + '/'

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
                        num_frames = (str(30 * config.lim_secs) if S_id in subjects_new_sys else str(15 * config.lim_secs))
                    else:
                        num_frames = None
                    with cd(config.AP_dir):
                        # Run AlphaPose on the data
                        ap_cmd = "python3 {} --cfg \"{}\" --checkpoint \"{}\" ".format("scripts/demo_inference.py", 
                                                                              "configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml", 
                                                                              "pretrained_models/halpe26_fast_res50_256x192.pth")
                        ap_cmd += "--sp --debug --gpu 0,1 --detbatch 1 --posebatch 30 --qsize 128 "
                        # ap_cmd += "--debug True --qsize 512 --posebatch 32 "
                        if config.limit_num_frames :
                            ap_cmd += "--maxframes {} ".format(num_frames)
                        ap_cmd += "--video \"{}\" --outdir \"{}\"".format(in_path, out_path)
                        print("\nTrying AP cmd: ", ap_cmd, end='\n\n')
                        os.system(ap_cmd)
                        
                        # Reformat the output json file
                        if config.save_preds:
                            mv_cmd = "mv {}alphapose-results.json {}CH{}_alphapose-results.json".format(out_path, out_path, ch)
                            print("\nTrying mv cmd: ", mv_cmd, end='\n\n')
                            os.system(mv_cmd)
                        else:
                            print("\nNot saving preds, so not moving the json file.")
                    print("Done with {}, CH_{}".format(S_id,ch))

def compile_JSON(config):
    '''
    Builds a combined task dataset (JSON) of 2D pose predictions from per-subject-per-channel alphapose pred JSONs
    '''
    print("\nCompiling predictions into JSON...")
    kpts_dict = {}
    for S_id, id in subjects_ALL_id_dict.items():
        if S_id in config.subjs_to_compile:
            print("\n--- {}: ---".format(S_id))
            # Construct paths
            in_json_path = config.dataset_path + str(id) + '/' + config.updrs_task + '/'

            # TODO: MAKE THIS NOT HARDCODED TO FIRST AND SECOND CHANNEL ENTRIES
            ch1_in_json_pth =  in_json_path + 'CH' + config.chs[0] + "_alphapose-results.json"
            ch2_in_json_pth =  in_json_path + 'CH' + config.chs[1] + "_alphapose-results.json"

            if not os.path.exists(ch1_in_json_pth):
                print("  ERR Input JSON not found: ", ch1_in_json_pth)
            elif not os.path.exists(ch2_in_json_pth):
                print("  ERR Input JSON not found: ", ch2_in_json_pth)
            else:
                # FOR NOW MUST USE A PAIR OF CHANNELS
                kpts_dict = filter_alphapose_results(in_json_path, S_id, config.updrs_task, config.chs, kpts_dict)
                print("  Successfully loaded alphapose data.")
    # Pickle the 2d keypoints dict
    if config.save_JSON:
        print("\nSaving to ", config.JSON_dataset_outpath)
        with open(config.JSON_dataset_outpath, 'wb') as handle:
            pickle.dump(kpts_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Successfully created 2D keypoint dataset.")
    else:
        print("\nNot saving 2D keypoint dataset.")

def get_config():
    config = SimpleNamespace()
    # Tasks
    config.get_2d_preds = False  # Proces videos to get 2d preds?
    config.compile_JSON = True  # Compile 2d preds into JSON dataset file?

    # config.subjs_to_get_preds = subjects_All
    config.subjs_to_get_preds = subjects_new_sys
    # config.subjs_to_get_preds = ["S29"]
    # config.subjs_to_compile = ['S01', 'S28', 'S29', 'S31']
    # config.subjs_to_compile = ['S01']
    # config.subjs_to_compile = ['S28']
    # config.subjs_to_compile = ['S29']
    # config.subjs_to_compile = ['S31']
    config.subjs_to_compile = subjects_new_sys
    
    # Settings
    config.save_preds = False        # Save the 2d preds?
    config.save_JSON = True         # Save the compiled JSON dataset file?

    config.limit_num_frames = True  # DEBUG: limit number of frames to process
    config.lim_secs = 10            # Mohsen used 2 min per video 

    # config.updrs_task = "free_form_oval"
    config.updrs_task = "tug_stand_walk_sit"
    # config.chs = ["003", "004"]   # Free Oval
    config.chs = ["006", "007"]     # TUG

    # Paths
    config.root_dir = os.path.dirname(os.path.realpath(__file__))
    config.videos_path = "/mnt/CAMERA-data/CAMERA/CAMERA visits/Mobility Visit/Study Subjects/"
    config.dataset_path = "/mnt/CAMERA-data/CAMERA/Other/lbidulka_dataset/"
    config.AP_dir = config.root_dir + "/AlphaPose"
    config.JSON_dataset_outpath = config.root_dir + "/data/body/2d_proposals/" + config.updrs_task + "_2D_kpts-DEBUG.pickle"
    return config

def main():
    config = get_config()

    if config.get_2d_preds:
        get_AlphaPoses(config)
    if config.compile_JSON:
        compile_JSON(config)

if __name__ == '__main__':
    main()

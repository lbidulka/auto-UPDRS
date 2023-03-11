import torch
import cv2
import argparse
from utils import info, post_processing, helpers, pose_visualization, metrics
from sklearn.metrics import accuracy_score
import numpy as np
from data.body.body_dataset import body_ts_loader, get_2D_keypoints_dict
import models.body_pose as body_nets


# Full body tracking tasks (e.g. gait analysis)
def body_tasks(input_args):
    # Fix the model setup by only saving the state_dict if needed
    # helpers.fix_model_setup(input_args.models_path + 'body_pose/Mohsens/model_lifter.pt', 
    #                 input_args.models_path + 'body_pose/model_lifter.pt')

    # Test out pose extractor on alphapose preds data
    subjects = ['9769']
    tasks = ['free_form_oval'] #['tug_stand_walk_sit']
    channels = [3]
    frame= 3 *2  # *2 because from CH3 we are also detecting the evaluator in each frame

    body_2D_proposals = get_2D_keypoints_dict(input_args.data_path, tasks=tasks, channels=channels, frame=frame)
    kpts_2D = torch.as_tensor(body_2D_proposals[subjects[0]][tasks[0]]['pos'][channels[0]], dtype=torch.float).unsqueeze(0)
    conf_2D = torch.as_tensor(body_2D_proposals[subjects[0]][tasks[0]]['conf'][channels[0]], dtype=torch.float).unsqueeze(0)

    body_3Dpose_lifter = body_nets.Lifter()
    body_3Dpose_lifter.load_state_dict(torch.load(input_args.models_path + 'body_pose/model_lifter.pt'))
    body_3Dpose_lifter.eval()
    pred_kpts_3D, pred_cam_angles = body_3Dpose_lifter(kpts_2D, conf_2D)

    kpts_2D = kpts_2D.detach().numpy()
    pred_kpts_3D = pred_kpts_3D.detach().numpy()
    pred_cam_angles = pred_cam_angles.detach().numpy()

    
    # Project back from canonical camera space to original camera space, then visualize
    kpts_3d_camspace = np.matmul(cv2.Rodrigues(pred_cam_angles[0])[0], pred_kpts_3D.reshape(-1, 3, 15))
    # need to swap the L and R legs for some reason... TODO: FIND OUT IF LIFTER OUTPUT ORDER IS AS INTENDED
    kpts_3d_camspace[:, :, 1:4], kpts_3d_camspace[:, :, 4:7] = kpts_3d_camspace[:, :, 4:7], kpts_3d_camspace[:, :, 1:4].copy()

    pose_visualization.visualize_pose(kpts_3d_camspace[0], kpts_2D=kpts_2D[0], num_dims=3, save_fig=True, out_fig_path="./auto_UPDRS/outputs/")
    pose_visualization.visualize_reproj(kpts_3d_camspace[0], kpts_2D[0], save_fig=True, out_fig_path="./auto_UPDRS/outputs/")
    
    # Load "free_form_oval" extracted 3D keypoints timeseries
    # ts_path = input_args.data_path + 'body/time_series/outputs_finetuned/'
    # gait_plots_outpath = input_args.output_path + 'plots/'
    # gait_loader = body_ts_loader(ts_path)   # All subjects
    # gait_processor = post_processing.gait_processor(gait_loader, gait_plots_outpath)

    # gait_processor.plot_feats_ts(show=False)

    # Test out some feature voting strategies
    # helpers.naive_voting(gait_processor)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_path", default="./auto_UPDRS/model_checkpoints/", help="model checkpoints path", type=str)
    parser.add_argument("--data_path", default="/mnt/CAMERA-data/CAMERA/Other/lbidulka_dataset/", help="input data path", type=str)
    parser.add_argument("--output_path", default="./auto_UPDRS/outputs/", help="output data path", type=str)
    return parser.parse_args()

def main():
    input_args = get_args()

    body_tasks(input_args)    
    
if __name__ == '__main__':
    main()

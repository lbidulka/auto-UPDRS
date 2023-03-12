import torch
import cv2
import argparse
from utils import info, post_processing, helpers, pose_visualization, metrics
from sklearn.metrics import accuracy_score
import numpy as np
from data.body.body_dataset import body_ts_loader, get_2D_keypoints_dict
import models.body_pose as body_nets
from pytorch3d.transforms import so3_exponential_map as rodrigues


# Fix the model setup by only saving the state_dict if needed
# helpers.fix_model_setup(input_args.models_path + 'body_pose/Mohsens/model_lifter.pt', 
#                 input_args.models_path + 'body_pose/model_lifter.pt')

def investigate_mohsens_loader():
    keypoints_PD = np.load('./PD_Gait_labeling/data/data_PD.npz', allow_pickle=True)
    keypoints_PD = keypoints_PD['positions_2d'].item()  # contains all subjects

    for subject in keypoints_PD.keys():
        for action in keypoints_PD[subject]:
            # print("\naction: ", action)
            for cam_idx in range(len(keypoints_PD[subject][action]['pos'])):
                # print("cam_idx: ", cam_idx)
                # Normalize camera frame
                kps = keypoints_PD[subject][action]['pos'][cam_idx]     # (n_frames, n_kpts, xy)
                conf = keypoints_PD[subject][action]['conf'][cam_idx]
                kps = kps[:,[0,1,2,3,4,5,6,8,9,10,11,12,13,14,15],:]
                conf = conf[:,[0,1,2,3,4,5,6,8,9,10,11,12,13,14,15],:]
                # print(kps.shape)
                # print(kps[0])
                kps = kps-kps[:,:1,:]   # centred on hip
                # print(kps.shape)
                # print("\ncentred: \n", kps[0])
                kps = np.transpose(kps,[0,2,1])
                # print(kps.shape)
                # print(kps[0])
                kps = kps.reshape(-1, 15*2)  # flat: (1, xxx ... yyy)
                # print(kps.shape)
                # print(kps[0])
                kps /= np.linalg.norm(kps,ord=2,axis=1,keepdims=True)
                # print("\nnormed: \n", kps[0])
                keypoints_PD[subject][action]['pos'][cam_idx] = kps
                keypoints_PD[subject][action]['conf'][cam_idx] = conf
                # break
            # break
        # break
    
    def fetch_train(subjects, actions, tag='train'):
        out_subject = []
        out_poses_2d = []
        out_confidences=[]
        for i in range(2):
            out_poses_2d.append([])
            out_confidences.append([])
        for subject in subjects:
            for action in actions:
                poses_2d = keypoints_PD[subject][action]['pos']
                conf_2d = keypoints_PD[subject][action]['conf']
                for i in range(2): # Iterate across cameras
                    out_poses_2d[i].append(poses_2d[i])
                    out_confidences[i].append(conf_2d[i])
                out_subject.append(np.ones(len(poses_2d[0]))*int(subject[1:]))
        for i in range(len(poses_2d)): # Iterate across cameras
            out_poses_2d[i]=np.concatenate(out_poses_2d[i],axis=0)        
            out_confidences[i]=np.concatenate(out_confidences[i],axis=0) 
        out_subject=np.concatenate(out_subject,axis=0)
        return  out_poses_2d, out_confidences, out_subject

    poses_2d_train, conf_2d_train, out_subject = fetch_train(['S01','S02','S25','S27','S28','S29'], ['WalkingOval'])
    print(len(poses_2d_train))
    print(poses_2d_train[0])
    print(poses_2d_train[0].shape)

# Full body tracking tasks (e.g. gait analysis)
def body_tasks(input_args):

    # ----------------------------------------------
    # Checkout Mohsens prediction data
    # ----------------------------------------------
    # investigate_mohsens_loader()

    # ----------------------------------------------
    # Test out pose extractor on alphapose preds data
    # ----------------------------------------------
    subjects = ['9769']
    tasks = ['free_form_oval'] #['tug_stand_walk_sit']
    channels = [3]
    norm_cam = True    # TODO: FIX THIS NORMING BUSINESS MOHSEN USED

    body_2D_proposals = get_2D_keypoints_dict(input_args.data_path, tasks=tasks, channels=channels, norm_cam=norm_cam)
    kpts_2D = torch.as_tensor(body_2D_proposals[subjects[0]][tasks[0]]['pos'][channels[0]], dtype=torch.float)
    conf_2D = torch.as_tensor(body_2D_proposals[subjects[0]][tasks[0]]['conf'][channels[0]], dtype=torch.float)

    body_3Dpose_lifter = body_nets.Lifter()
    body_3Dpose_lifter.load_state_dict(torch.load(input_args.models_path + 'body_pose/model_lifter.pt'))
    body_3Dpose_lifter.eval()

    with torch.no_grad():
        pred_kpts_3D, pred_cam_angles = body_3Dpose_lifter(kpts_2D, conf_2D)

    kpts_2D = kpts_2D.detach().numpy()
    pred_kpts_3D = pred_kpts_3D 
    pred_cam_angles = pred_cam_angles 

    # Project back from canonical camera space to original camera space 
    kpts_3d_camspace = rodrigues(pred_cam_angles)[0] @ pred_kpts_3D.reshape(-1, 3, 15)

    rot_poses = torch.transpose(kpts_3d_camspace, 2, 1) # swap to do procrustes
    rot_poses -= rot_poses[:, :1]   # center the poses on hip
    pred_aligned = metrics.procrustes_torch(rot_poses[0:1], rot_poses)  # Aligns to first pose?
    pred_aligned = np.transpose(pred_aligned, [0, 2, 1])  # swap back

    # need to swap the L and R legs for some reason... TODO: FIND OUT IF LIFTER OUTPUT ORDER IS AS INTENDED
    pred_aligned[:, :, 1:4], pred_aligned[:, :, 4:7] = pred_aligned[:, :, 4:7], pred_aligned[:, :, 1:4].copy()

    # Visualize the pose results
    pose_visualization.visualize_pose(pred_aligned[0], kpts_2D=kpts_2D[0], save_fig=True, out_fig_path="./auto_UPDRS/outputs/", normed_in=norm_cam)
    pose_visualization.visualize_reproj(pred_aligned[0], kpts_2D[0], save_fig=True, out_fig_path="./auto_UPDRS/outputs/")

    # Output 2D & 3D keypoints, so we can make a video
    np.save("./auto_UPDRS/outputs/3D_kpts.npy", pred_aligned)
    # np.save("./auto_UPDRS/outputs/vids_2d/2D_kpts.npy", kpts_2D)
    # pose_visualization.pose2d_video(kpts_2D, outpath="./auto_UPDRS/outputs/vids_2d/")
    pose_visualization.pose3d_video(pred_aligned, outpath="./auto_UPDRS/outputs/vids_3d/")

    # ----------------------------------------------
    # Load "free_form_oval" extracted 3D keypoints timeseries
    # ----------------------------------------------
    
    # ts_path = './auto_UPDRS/data/body/time_series/outputs_finetuned/'
    # gait_plots_outpath = input_args.output_path + 'plots/'
    # gait_loader = body_ts_loader(ts_path)   # All subjects
    # gait_processor = post_processing.gait_processor(gait_loader, gait_plots_outpath)

    # print(len(gait_processor.data_normal))
    # print(gait_processor.data_normal[0].shape)
    # print(gait_processor.data_normal[0][0])

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

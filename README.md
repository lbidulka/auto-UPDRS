# !!! (Course Project) Self-Supervised 3D Human Pose Estimation for Parkinson’s Disease and Beyond !!!

This is my cleaned up branch for the course project material only.

## Data:
The original videos cannot be shared due to privacy concerns, but I have uploaded the extracted keypoints:
- 'data/body/2d_proposals/tug_stand_walk_sit_CH006_CH007_2D_kpts.pickle' contains the filtered 2D backbone pose predictions (MVP-3D lifter training dataset)
- 'data/body/3d_time_series/CH006_tug_stand_walk_sit.pickle' contains the MVP-3D predictions on the TUG data

## MVP-3D Transfer to TUG data:
- create_PD_2d_dataset.py uses AlphaPose to get the 2D backbone poses on the UPDRS video data and/or filters the 2D poses to create the MVP-3D lifter training dataset
    - 'data/body/body_dataset.py' contains the code for filtering the 2D poses (filter_alphapose_results() and filter_ap_detections()) 
    - 'utils/alphapose_filtering.py' contains the pixel-space filter definitions
- train_PD_bodylifter.py trains MVP-3D on the desired UPDRS task (Oval or TUG) and also produces the MVP-3D TUG data predictions

## UncertNet Correction Network:

- 'uncertnet_experiments.py' contains high level control for training and evaluating UncertNet
- 'uncertnet/uncertnet.py' defines the UncerNet and its training
- 'uncertnet/dataset.py' helps load and handle data for the UncertNet
- 'data/body/h36m/uncertnet' contains the numpy data files for training and testing

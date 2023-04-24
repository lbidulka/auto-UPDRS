# !!! (Course Project) Self-Supervised 3D Human Pose Estimation for Parkinson’s Disease and Beyond !!!

This is my cleaned up branch to properly represent the final course project. No code has been modified, but some code which became irrelevant has been removed, files have been renamed for clarity, and this README has been updated to be more informative.

Main branch is left as it was at the time of submission for transparency. 

## Main Files:
---
### **MVP-3D transfer to TUG data:**
- 'MVP-3D_create_2d_dataset.py' 
    - gets the 2D backbone (AlphaPose) predictions on UPDRS video data 
    - filters 2-view 2D poses to create the MVP-3D training dataset
- 'MVP-3D_train_and_eval.py'
    - trains MVP-3D on the desired UPDRS task (Oval or TUG) 
    - produces output MVP-3D predictions for evaluation/plotting
### **UncertNet Correction of MVP-3D on Human 3.6M:**
- 'uncertnet_create_dataset.py' 
    - uses MVP-3D to create the uncertnet training dataset from Human 3.6M
    - is a bit clunky and not fully consistent with the MVP-3D transfer format since it was done earlier, but it works
    - follow description in "Data" section below to use it correctly to generate the UncertNet dataset
- 'uncertnet_experiments.py' 
    - provides high level control for training and evaluating UncertNet
- 'uncertnet/uncertnet.py' 
    - defines UncertNet and its training
- 'uncertnet/dataset.py' 
    - helps load and handle data for UncertNet

## Data:
---
The original UPDRS task videos cannot be shared due to privacy concerns, but I have provided the following:
### **MVP-3D transfer to TUG data:**
- 'data/body/2d_proposals/tug_stand_walk_sit_CH006_CH007_2D_kpts.pickle' 
    - contains the filtered 2D backbone pose predictions, for training MVP-3D
- 'data/body/3d_time_series/CH006_tug_stand_walk_sit.pickle' 
    - contains the MVP-3D predictions on the TUG data after training, for evaluation and plotting
### **UncertNet Correction of MVP-3D on Human 3.6M:**
- As mentioned above, due to bad design choices and limited time, this is done like so:
    - clone the original MVP-3D repo: [PD_Gait_labeling](https://github.com/mgholamikn/PD_Gait_labeling)
    - As described in the example Google Colab of [PD_Gait_labeling](https://github.com/mgholamikn/PD_Gait_labeling), download the data and pre-trained models
    - copy this 'uncertnet_create_dataset.py' script to the main directory of [PD_Gait_labeling](https://github.com/mgholamikn/PD_Gait_labeling) and run it there to produce the .npy files for training uncertnet
    - copy all the .npy files into this repo under 'data/body/h36m/uncertnet'


<!-- ## MVP-3D Transfer to TUG data:
- create_PD_2d_dataset.py uses AlphaPose to get the 2D backbone poses on the UPDRS video data and/or filters the 2D poses to create the MVP-3D lifter training dataset
    - 'data/body/body_dataset.py' contains the code for filtering the 2D poses (filter_alphapose_results() and filter_ap_detections()) 
    - 'utils/alphapose_filtering.py' contains the pixel-space filter definitions
- train_PD_bodylifter.py trains MVP-3D on the desired UPDRS task (Oval or TUG) and also produces the MVP-3D TUG data predictions -->
<!-- 
## UncertNet Correction Network:

- 'uncertnet_experiments.py' contains high level control for training and evaluating UncertNet
- 'uncertnet/uncertnet.py' defines the UncerNet and its training
- 'uncertnet/dataset.py' helps load and handle data for the UncertNet
- 'data/body/h36m/uncertnet' contains the numpy data files for training and testing -->

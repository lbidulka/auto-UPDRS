import numpy as np
from utils import pose_visualization

# Very simple stuff
kpts_2D = np.load("./auto_UPDRS/outputs/vids_2d/2D_kpts.npy")
pose_visualization.pose2d_video(kpts_2D, outpath="./auto_UPDRS/outputs/vids_2d/")
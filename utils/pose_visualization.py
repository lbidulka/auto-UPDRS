import matplotlib.pyplot as plt
import numpy as np


def plot_3D_skeleton(kpts_3D, ax, colours=['r', 'g', 'b'], linewidth=0.5):
    '''
    Plot the 3D skeleton on the given axis

    args:
        kpts_3D: 3D keypoints (3, 15), xyz by 15 keypoints

            0:  pelvis
            1:  right hip
            2:  right knee
            3:  right ankle
            4:  left hip
            5:  left knee
            6:  left ankle
            7:  neck
            8:  head
            9:  right shoulder
            10: right elbow
            11: right hand
            12: left shoulder
            13: left elbow
            14: left hand
    '''
    # Skeleton sequences for 3D plotting
    LA = [7, 12, 13, 14] # L Arm: Neck, LShoulder, LElbow, LWrist
    RA = [7, 9, 10, 11]  # R Arm: Neck, RShoulder, RElbow, RWrist
    LL = [0, 4, 5, 6]    # L Leg: Hip, LHip, LKnee, LAnkle
    RL = [0, 1, 2, 3]    # R Leg: Hip, RHip, RKnee, RAnkle
    T = [8, 7, 0]        # Torso: Head, Neck, Hip

    ax.plot(kpts_3D[0, LA], kpts_3D[1, LA], kpts_3D[2, LA], color=colours[2], linewidth=linewidth) 
    ax.plot(kpts_3D[0, RA], kpts_3D[1, RA], kpts_3D[2, RA], color=colours[0], linewidth=linewidth)
    ax.plot(kpts_3D[0, LL], kpts_3D[1, LL], kpts_3D[2, LL], color=colours[2], linewidth=linewidth)
    ax.plot(kpts_3D[0, RL], kpts_3D[1, RL], kpts_3D[2, RL], color=colours[0], linewidth=linewidth)
    ax.plot(kpts_3D[0, T], kpts_3D[1, T], kpts_3D[2, T], color=colours[1], linewidth=linewidth)

def plot_2D_skeleton(kpts_2D, ax, proj_plot=True, z_off=0, zdir='z', colours=['r', 'g', 'b'], linewidth=0.5):
    '''
    Plot the 2D skeleton on the given axis.

    args:
        kpts_2D: 2D keypoints (30), xy xy xy ...
        proj_plot: if True, plot on 3D axis, else plot on 2D axis
    '''
    # Skeleton sequences for 2D plotting
    LA = [26, 0, 4, 8]    # L Arm: Neck, LShoulder, LElbow, LWrist
    RA = [26, 2, 6, 10]   # R Arm: Neck, RShoulder, RElbow, RWrist
    LL = [28, 12, 16, 20] # L Leg: Hip, LHip, LKnee, LAnkle
    RL = [28, 14, 18, 22] # R Leg: Hip, RHip, RKnee, RAnkle
    T = [24, 26, 28]      # Torso: Head, Neck, Hip

    # For plotting on 3D axis
    if proj_plot:
        ax.plot(kpts_2D[LA], kpts_2D[[x+1 for x in LA]], zs=z_off, zdir=zdir, color=colours[2], linewidth=linewidth)
        ax.plot(kpts_2D[RA], kpts_2D[[x+1 for x in RA]], zs=z_off, zdir=zdir, color=colours[0], linewidth=linewidth)
        ax.plot(kpts_2D[LL], kpts_2D[[x+1 for x in LL]], zs=z_off, zdir=zdir, color=colours[2], linewidth=linewidth)
        ax.plot(kpts_2D[RL], kpts_2D[[x+1 for x in RL]], zs=z_off, zdir=zdir, color=colours[0], linewidth=linewidth)
        ax.plot(kpts_2D[T], kpts_2D[[x+1 for x in T]], zs=z_off, zdir=zdir, color=colours[1], linewidth=linewidth)
    # For plotting on 2D axis
    else:
        ax.plot(kpts_2D[LA], kpts_2D[[x+1 for x in LA]], color=colours[2], linewidth=linewidth)
        ax.plot(kpts_2D[RA], kpts_2D[[x+1 for x in RA]], color=colours[0], linewidth=linewidth)
        ax.plot(kpts_2D[LL], kpts_2D[[x+1 for x in LL]], color=colours[2], linewidth=linewidth)
        ax.plot(kpts_2D[RL], kpts_2D[[x+1 for x in RL]], color=colours[0], linewidth=linewidth)
        ax.plot(kpts_2D[T], kpts_2D[[x+1 for x in T]], color=colours[1], linewidth=linewidth)

# TODO: ADD VISUALIZATION OF 3D POSE PROJECTION ONTO 2D INPUT POSE
def visualize_pose(kpts_3D=None, kpts_2D=None, num_dims=2, save_fig=False, show_fig=False, out_fig_path=None):
    '''
    Visualization of the estimated pose
    
    input keypoint          idx (x,y)
    ----------------------  ---------
        {0,  "LShoulder"},  -> 0, 1
        {1,  "RShoulder"},  -> 2, 3
        {2,  "LElbow"},     -> 4, 5
        {3,  "RElbow"},     -> 6, 7
        {4,  "LWrist"},     -> 8, 9
        {5, "RWrist"},      -> 10, 11
        {6, "LHip"},        -> 12, 13
        {7, "RHip"},        -> 14, 15
        {8, "LKnee"},       -> 16, 17
        {9, "Rknee"},       -> 18, 19
        {10, "LAnkle"},     -> 20, 21
        {11, "RAnkle"},     -> 22, 23
        {12,  "Head"},      -> 24, 25
        {13,  "Neck"},      -> 26, 27
        {14,  "Hip"},       -> 28, 29

    Args:
        kpts_3D: 3D keypoints (3, 15), xyz by 15 keypoints
        kpts_2D: 2D keypoints (30), xy xy xy ...
        num_dims: 2 or 3, if 2 then just plot 2D keypoints, if 3 then plot 3D keypoints and 2D keypoints
    '''
    if kpts_2D is None and kpts_3D is None:
        print("ERROR: No keypoints to visualize!")
        return
    # How many of kpts_2D and kpts_3D are not None? (Lazy way, but works)
    nrows = 0
    if kpts_2D is not None:
        nrows += 1
    if kpts_3D is not None:
        nrows += 1

    fig = plt.figure(layout='constrained')
    if num_dims == 3:
        if kpts_2D is not None:
            ax = fig.add_subplot(nrows, 1, 1)
            plot_2D_skeleton(kpts_2D, ax, proj_plot=False)
            for i in range(0, len(kpts_2D), 2):
                # print(i, " xy: ", kpts_2D[i], kpts_2D[i+1])
                ax.scatter(kpts_2D[i], kpts_2D[i+1], color='b', s=1)
            ax.set_title('Subject 2D Pose')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_xlim(0,3840)
            ax.set_ylim(2160,0)
        if kpts_3D is not None:
            ax = fig.add_subplot(nrows, 1, 2, projection='3d')
            plot_3D_skeleton(kpts_3D, ax)
            for i in range(kpts_3D.shape[1]):
                # print("xyz: ", kpts_3D[0][i], kpts_3D[1][i], kpts_3D[2][i])
                ax.scatter(kpts_3D[0][i], kpts_3D[1][i], kpts_3D[2][i], color='b', s=1)

            ax.view_init(elev=30., azim=35.)
            ax.set_title('Subject 3D Pose')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
    
    elif num_dims == 2:
        ax = fig.add_subplot()
        if kpts_2D is not None:
            plot_2D_skeleton(kpts_2D, ax, proj_plot=False,)
            for i in range(0, len(kpts_2D), 2):
                # print(i, " xy: ", kpts_2D[i], kpts_2D[i+1])
                ax.scatter(kpts_2D[i], kpts_2D[i+1], color='b', s=1)
        ax.set_title('Subject 2D Pose')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim(0,3840)
        ax.set_ylim(2160,0)

    # Show/Save the figure
    if (save_fig and (out_fig_path is not None)): plt.savefig(out_fig_path + 'pose.png', dpi=500)
    if show_fig: plt.show()
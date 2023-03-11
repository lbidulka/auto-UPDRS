import matplotlib.pyplot as plt
import numpy as np


def plot_3D_skeleton(kpts_3D, ax, colours=['r', 'g', 'b'], linewidth=0.5, dims=3, 
                     plt_scatter=False, scatter_size=1, scatter_colour='r'):
    '''
    Plot the 3D skeleton on the given axis, in 3D or with weak projection to 2D

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

        dims: 2 or 3, resulting in 3D plot (normal) or 2D plot (weak perspective projection by ignoring depth)
        plt_scatter: if True, also plot the keypoints on top of the skeleton
    '''
    # Skeleton sequences for 3D plotting
    LA = [7, 12, 13, 14] # L Arm:  Neck, LShoulder, LElbow, LWrist
    RA = [7, 9, 10, 11]  # R Arm:  Neck, RShoulder, RElbow, RWrist
    LL = [0, 4, 5, 6]    # L Leg:  Hip, LHip, LKnee, LAnkle
    RL = [0, 1, 2, 3]    # R Leg:  Hip, RHip, RKnee, RAnkle
    T = [8, 7, 0]        # Torso:  Head, Neck, Hip

    # normal, note that y and z axes are swapped to better compare to 2D plots
    if dims == 3:
        ax.plot(kpts_3D[0, LA], kpts_3D[2, LA], kpts_3D[1, LA], color=colours[2], linewidth=linewidth) 
        ax.plot(kpts_3D[0, RA], kpts_3D[2, RA], kpts_3D[1, RA], color=colours[0], linewidth=linewidth)
        ax.plot(kpts_3D[0, LL], kpts_3D[2, LL], kpts_3D[1, LL], color=colours[2], linewidth=linewidth)
        ax.plot(kpts_3D[0, RL], kpts_3D[2, RL], kpts_3D[1, RL], color=colours[0], linewidth=linewidth)
        ax.plot(kpts_3D[0, T], kpts_3D[2, T], kpts_3D[1, T], color=colours[1], linewidth=linewidth)
        if plt_scatter:
            ax.scatter(kpts_3D[0, :], kpts_3D[2, :], kpts_3D[1, :], s=scatter_size, c=scatter_colour)
    # weak perspective projection
    elif dims == 2:
        ax.plot(kpts_3D[0, LA], kpts_3D[1, LA], color=colours[2], linewidth=linewidth) 
        ax.plot(kpts_3D[0, RA], kpts_3D[1, RA], color=colours[0], linewidth=linewidth)
        ax.plot(kpts_3D[0, LL], kpts_3D[1, LL], color=colours[2], linewidth=linewidth)
        ax.plot(kpts_3D[0, RL], kpts_3D[1, RL], color=colours[0], linewidth=linewidth)
        ax.plot(kpts_3D[0, T], kpts_3D[1, T], color=colours[1], linewidth=linewidth)
        if plt_scatter:
            ax.scatter(kpts_3D[0, :], kpts_3D[1, :], s=scatter_size, c=scatter_colour)

def plot_2D_skeleton(kpts_2D, ax, proj_plot=True, z_off=0, zdir='z', colours=['r', 'g', 'b'], linewidth=0.5, 
                     plt_scatter=False, scatter_size=0.5, scatter_colour='b'):
    '''
    Plot the 2D skeleton on the given axis.

    args:
        kpts_2D: 2D keypoints (30), xy xy xy ...
        proj_plot: if True, plot on 3D axis, else plot on 2D axis                 # Not working yet, low priority
        plt_scatter: if True, also plot the keypoints on top of the skeleton
    '''
    # Skeleton sequences for 2D plotting
    all_pts = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]
    LA = [26, 0, 4, 8]    # L Arm:  Neck, LShoulder, LElbow, LWrist
    RA = [26, 2, 6, 10]   # R Arm:  Neck, RShoulder, RElbow, RWrist
    LL = [28, 12, 16, 20] # L Leg:  Hip, LHip, LKnee, LAnkle
    RL = [28, 14, 18, 22] # R Leg:  Hip, RHip, RKnee, RAnkle
    T = [24, 26, 28]      # Torso:  Head, Neck, Hip

    # For plotting on 3D axis
    if proj_plot:
        ax.plot(kpts_2D[LA], kpts_2D[[x+1 for x in LA]], zs=z_off, zdir=zdir, color=colours[2], linewidth=linewidth)
        ax.plot(kpts_2D[RA], kpts_2D[[x+1 for x in RA]], zs=z_off, zdir=zdir, color=colours[0], linewidth=linewidth)
        ax.plot(kpts_2D[LL], kpts_2D[[x+1 for x in LL]], zs=z_off, zdir=zdir, color=colours[2], linewidth=linewidth)
        ax.plot(kpts_2D[RL], kpts_2D[[x+1 for x in RL]], zs=z_off, zdir=zdir, color=colours[0], linewidth=linewidth)
        ax.plot(kpts_2D[T], kpts_2D[[x+1 for x in T]], zs=z_off, zdir=zdir, color=colours[1], linewidth=linewidth)
        # TODO: GET 3D SCATTER PLOT ON 2D SUB-AX WORKING (LOW PRIORITY)
        if plt_scatter:
            ax.scatter(kpts_2D[all_pts], kpts_2D[[x+1 for x in all_pts]], 
                       zs=z_off, zdir=zdir, s=scatter_size, c=scatter_colour)
    # For plotting on 2D axis
    else:
        ax.plot(kpts_2D[LA], kpts_2D[[x+1 for x in LA]], color=colours[2], linewidth=linewidth)
        ax.plot(kpts_2D[RA], kpts_2D[[x+1 for x in RA]], color=colours[0], linewidth=linewidth)
        ax.plot(kpts_2D[LL], kpts_2D[[x+1 for x in LL]], color=colours[2], linewidth=linewidth)
        ax.plot(kpts_2D[RL], kpts_2D[[x+1 for x in RL]], color=colours[0], linewidth=linewidth)
        ax.plot(kpts_2D[T], kpts_2D[[x+1 for x in T]], color=colours[1], linewidth=linewidth)
        if plt_scatter:
            ax.scatter(kpts_2D[all_pts], kpts_2D[[x+1 for x in all_pts]], 
                       s=scatter_size, c=scatter_colour)

def visualize_pose(kpts_3D=None, kpts_2D=None, num_dims=2, save_fig=False, show_fig=False, out_fig_path=None):
    '''
    Visualization of the estimated pose
    
    input 2D keypoint          idx (x,y)
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

    fig = plt.figure(layout='constrained', )
    if num_dims == 3:
        if kpts_2D is not None:
            ax = fig.add_subplot(nrows, 1, 1)
            plot_2D_skeleton(kpts_2D, ax, proj_plot=False, plt_scatter=True)
            ax.set_title('Subject 2D Pose')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_xlim(0,3840)
            ax.set_ylim(2160,0)
        if kpts_3D is not None:
            ax = fig.add_subplot(nrows, 1, 2, projection='3d')
            plot_3D_skeleton(kpts_3D, ax, plt_scatter=True)
            ax.view_init(elev=20., azim=-110.)
            ax.set_title('Subject 3D Pose')
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_zlabel('Y')      
            ax.set_xlim(min(kpts_3D[0]), max(kpts_3D[0]*2))
            ax.set_zlim(max(kpts_3D[1]), min(kpts_3D[1]))
    
    elif num_dims == 2:
        ax = fig.add_subplot()
        if kpts_2D is not None:
            plot_2D_skeleton(kpts_2D, ax, proj_plot=False, plt_scatter=True)
        ax.set_title('Subject 2D Pose')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_xlim(0,3840)
        ax.set_ylim(2160,0)

    # Show/Save the figure
    if (save_fig and (out_fig_path is not None)): plt.savefig(out_fig_path + 'pose.png', dpi=500, bbox_inches='tight')
    if show_fig: plt.show()

def visualize_reproj(kp_3D, kp_2D, save_fig=False, show_fig=False, out_fig_path=None):
    '''
    Visualize the reprojection of 3D lifter preds onto 2D backbone preds

    TODO: IMPLEMENT REAL REPROJ, INSTEAD OF WEAK REPROJ

    args:
        kpts_3D: 3D keypoints (3, 15), xyz by 15 keypoints
        kpts_2D: 2D keypoints (30), xy xy xy ...
    '''
    fig = plt.figure(layout='constrained')
    ax = fig.add_subplot()

    # Rescale the keypoints
    num_joints = 15
    scale_p2d = np.sqrt(np.square(kp_2D).sum() / num_joints*2)
    kp2d_scaled = kp_2D / scale_p2d

    scale_p3d = np.sqrt(np.square(kp_3D[:2, :]).sum() / num_joints*2)
    kp3d_scaled = kp_3D[:2, :] / scale_p3d

    plot_2D_skeleton(kp2d_scaled, ax, plt_scatter=True, proj_plot=False)
    plot_3D_skeleton(kp3d_scaled, ax, plt_scatter=True, dims=2)

    ax.set_title('Backbone and Lifter-reproj 2D Poses')
    ax.invert_yaxis()
    ax.set_xlim(min(kp3d_scaled[0]), max(kp3d_scaled[0]*2))

    # Show/Save the figure
    if (save_fig and (out_fig_path is not None)): plt.savefig(out_fig_path + 'pose-reproj.png', dpi=500, bbox_inches='tight')
    if show_fig: plt.show()
    
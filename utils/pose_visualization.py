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
                     plt_scatter=False, scatter_size=1.5, scatter_colour='b'):
    '''
    Plot the 2D skeleton on the given axis.

    args:
        kpts_2D: 2D keypoints (30), xxx ... yyy
        proj_plot: if True, plot on 3D axis, else plot on 2D axis                 # Not working yet, low priority
        plt_scatter: if True, also plot the keypoints on top of the skeleton
    '''
    # Skeleton sequences for 2D plotting
    all_pts = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    LA = [13, 0, 2, 4]  # L Arm:  Neck, LShoulder, LElbow, LWrist
    RA = [13, 1, 3, 5]  # R Arm:  Neck, RShoulder, RElbow, RWrist
    LL = [14, 6, 8, 10] # L Leg:  Hip, LHip, LKnee, LAnkle
    RL = [14, 7, 9, 11] # R Leg:  Hip, RHip, RKnee, RAnkle
    T = [12, 13, 14]    # Torso:  Head, Neck, Hip

    xs = kpts_2D[all_pts]
    ys = kpts_2D[[x + len(all_pts) for x in all_pts]]

    # For plotting on 3D axis
    if proj_plot:
        ax.plot(xs[LA], ys[LA], zs=z_off, zdir=zdir, color=colours[2], linewidth=linewidth)
        ax.plot(xs[RA], ys[RA], zs=z_off, zdir=zdir, color=colours[0], linewidth=linewidth)
        ax.plot(xs[LL], ys[LL], zs=z_off, zdir=zdir, color=colours[2], linewidth=linewidth)
        ax.plot(xs[RL], ys[RL], zs=z_off, zdir=zdir, color=colours[0], linewidth=linewidth)
        ax.plot(xs[T], ys[T], zs=z_off, zdir=zdir, color=colours[1], linewidth=linewidth)
        # TODO: GET 3D SCATTER PLOT ON 2D SUB-AX WORKING (LOW PRIORITY)
        if plt_scatter:
            ax.scatter(kpts_2D[all_pts], kpts_2D[[x + len(all_pts) for x in all_pts]], 
                       zs=z_off, zdir=zdir, s=scatter_size, c=scatter_colour)
    # For plotting on 2D axis
    else:
        ax.plot(xs[LA], ys[LA], color=colours[2], linewidth=linewidth)
        ax.plot(xs[RA], ys[RA], color=colours[0], linewidth=linewidth)
        ax.plot(xs[LL], ys[LL], color=colours[2], linewidth=linewidth)
        ax.plot(xs[RL], ys[RL], color=colours[0], linewidth=linewidth)
        ax.plot(xs[T], ys[T], color=colours[1], linewidth=linewidth)
        if plt_scatter:
            ax.scatter(xs, ys, 
                       s=scatter_size, c=scatter_colour)

def visualize_pose(kpts_3D=None, kpts_2D=None, num_dims=3, save_fig=False, show_fig=False, out_fig_path=None, normed_in=False):
    '''
    Visualization of the estimated pose
    
    input 2D keypoint          idx (x,y)
    ----------------------  ---------
        {0,  "LShoulder"},  -> 0, 15
        {1,  "RShoulder"},  -> 1, 16
        {2,  "LElbow"},     -> 2, 17
        {3,  "RElbow"},     -> 3, 18
        {4,  "LWrist"},     -> 4, 19
        {5, "RWrist"},      -> 5, 20
        {6, "LHip"},        -> 6, 21
        {7, "RHip"},        -> 7, 22
        {8, "LKnee"},       -> 8, 23
        {9, "Rknee"},       -> 9, 24
        {10, "LAnkle"},     -> 10, 25
        {11, "RAnkle"},     -> 11, 26
        {12,  "Head"},      -> 12, 27
        {13,  "Neck"},      -> 13, 28
        {14,  "Hip"},       -> 14, 29

    Args:
        kpts_3D: 3D keypoints (3, 15), xyz by 15 keypoints
        kpts_2D: 2D keypoints (30), xxx ... yyy
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
            ax.invert_yaxis()
            if not normed_in:
                ax.set_xlim(0,3840)
                ax.set_ylim(2160,0) 

        if kpts_3D is not None:
            ax = fig.add_subplot(nrows, 1, 2, projection='3d')
            plot_3D_skeleton(kpts_3D, ax, plt_scatter=True)
            if not normed_in:
                ax.view_init(elev=20., azim=-110.)
            else:
                ax.view_init(elev=20., azim=90.)
            ax.set_title('Subject 3D Pose')
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_zlabel('Y')
            ax.invert_zaxis()
            # ax.set_xlim(min(kpts_3D[0]), max(kpts_3D[0]*2))
            # ax.set_zlim(max(kpts_3D[1]), min(kpts_3D[1]))
    
    elif num_dims == 2:
        ax = fig.add_subplot()
        if kpts_2D is not None:
            plot_2D_skeleton(kpts_2D, ax, proj_plot=False, plt_scatter=True)
        ax.set_title('Subject 2D Pose')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.invert_yaxis()
        if not normed_in:
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

    plot_2D_skeleton(kp2d_scaled, ax, plt_scatter=True, proj_plot=False, scatter_size=20)
    plot_3D_skeleton(kp3d_scaled, ax, plt_scatter=True, dims=2, scatter_size=20)

    ax.set_title('Backbone and Lifter-reproj 2D Poses')
    ax.invert_yaxis()
    ax.set_xlim(min(kp3d_scaled[0]), max(kp3d_scaled[0]*2))

    # Show/Save the figure
    if (save_fig and (out_fig_path is not None)): plt.savefig(out_fig_path + 'pose-reproj.png', dpi=500, bbox_inches='tight')
    if show_fig: plt.show()

def pose2d_video():
    '''
    Create video of 2D pose predictions
    '''
    pass
    
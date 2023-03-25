import torch
import numpy as np
from pytorch3d.transforms import so3_exponential_map as rodrigues
from utils import metrics


def procrustes_torch(X, Y):
    """
    Reimplementation of MATLAB's `procrustes` function to Numpy.
    """
    X1=X[:,[0,1,4,7,8,9,10,13]]
    Y1=Y[:,[0,1,4,7,8,9,10,13]]
    batch,n, m = X1.shape
    batch, ny, my = Y1.shape

    muX = torch.mean(X1,dim=1,keepdim=True)
    muY = torch.mean(Y1,dim=1,keepdim=True)

    X0 = X1 - muX
    Y0 = Y1 - muY

    # optimum rotation matrix of Y
    A = torch.matmul(torch.transpose(X0,-1,-2), Y0)
    U,s,V = torch.svd(A)
    T = torch.matmul(V, torch.transpose(U,-1,-2))

    X1=X
    Y1=Y
    muX = torch.mean(X1,dim=1,keepdim=True)
    muY = torch.mean(Y1,dim=1,keepdim=True)

    X0 = X1 - muX
    Y0 = Y1 - muY

    Z = torch.matmul(Y0, T) + muX


    return np.array(Z.cpu())

def reshape_and_align(kpts_3d, angles, reproj=False, swap_legs=True):
    '''
    Unflattens the lifter output 3D keypoints, centres them on the hip, and procrustes aligns them to first pose

    args:
        kpts_3d: (B, 45) predicted 3D keypoints in canonical (world) space
        angles: (B, 3) predicted camera angles
        reproj: (bool) whether to reproject the keypoints back to the original camera space
        swap_legs: (bool) whether to swap the leg keypoints (TODO: TALK TO MOHSEN ABT WHY THIS IS NECESSARY)

    returns:
        kpts_3d: (B, XYZ, 15) reshaped 3D keypoints in canonical (world) or cam space
        rots: (B, 3) predicted camera angles, or (B, 3, 3) predicted rotation matrix
    '''
    kpts_3d = kpts_3d.reshape(-1, 3, 15)    # (B, XYZ, 15)

    if reproj:
        kpts_3d = rodrigues(angles)[0] @ kpts_3d.reshape(-1, 3, 15)
        angles = rodrigues(angles)[0]

    # Project back from canonical camera space to original camera space
    kpts_3d = torch.transpose(kpts_3d, 2, 1) # swap axes to do procrustes
    kpts_3d -= kpts_3d[:, :1]   # center the poses on hip
    kpts_3d_aligned = metrics.procrustes_torch(kpts_3d[0:1], kpts_3d)  # Aligns to first pose?
    kpts_3d_aligned = np.transpose(kpts_3d_aligned, [0, 2, 1])  # swap axes back

    # need to swap the L and R legs for some reason... TODO: FIND OUT IF LIFTER OUTPUT ORDER IS AS INTENDED
    if swap_legs:
        kpts_3d_aligned[:, :, 1:4], kpts_3d_aligned[:, :, 4:7] = kpts_3d_aligned[:, :, 4:7], kpts_3d_aligned[:, :, 1:4].copy()

    return kpts_3d_aligned, angles

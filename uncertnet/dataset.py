import torch
import numpy as np
from torch.utils.data import Dataset


class H36MDataset(Dataset):
    """Human3.6M dataset including images."""

    def __init__(self, poses_3d_triang, poses_3d_gt, poses_2d, confidences, subjects):
        self.poses_2d = poses_2d
        self.poses_3d_triang = poses_3d_triang
        self.poses_3d_gt = poses_3d_gt  
        self.subjects = subjects
        self.conf = confidences

    def __len__(self):
        return self.poses_2d[0].shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = dict()
        for c_idx in range(len(self.poses_2d)):
            p2d = torch.Tensor(self.poses_2d[c_idx][idx].astype('float32')).cuda()
            p3d_tr = torch.Tensor(self.poses_3d_triang[c_idx][idx].astype('float32')).cuda()
            p3d_gt = torch.Tensor(self.poses_3d_gt[c_idx][idx].astype('float32')).cuda()

            sample['cam' + str(c_idx)] = p2d
            sample['cam' + str(c_idx)+'_3d_tr'] = p3d_tr
            sample['cam' + str(c_idx)+'_3d_gt'] = p3d_gt

        sample['confidences'] = dict()
        for c_idx in range(len(self.poses_2d)):
            if self.conf is None:
                sample['confidences'][c_idx] = torch.ones(15) 
            else:
                sample['confidences'][c_idx] = torch.Tensor(self.conf[c_idx][idx].astype('float32')).cuda()
        sample['subjects'] = self.subjects[idx]

        return sample
    

class H36M():
    '''
    Loads Human3.6M data, splits into train/val/test, and creates data loaders
    '''
    def __init__(self, config):
        print("Loading data...", end=" ")
        data_cam_ids, data_ap_2d_poses, data_pred_poses, data_pred_rots, data_tr_poses, data_gt_poses = self.load_data(config, split='train')
        test_data_cam_ids, test_data_ap_2d_poses, test_data_pred_poses, test_data_pred_rots, test_data_tr_poses, test_data_gt_poses = self.load_data(config, split='test')
        print("Using cams: {}".format(config.cams))
        print("loaded: {} train samples, {} test samples".format(len(data_cam_ids), len(test_data_cam_ids)), "\n")
        splits = self.get_splits(config, data_cam_ids, data_ap_2d_poses, data_pred_poses, data_pred_rots, data_tr_poses, data_gt_poses)

        (train_cam_ids, train_ap_2d_poses, train_pred_poses, train_pred_rots, train_tr_poses, train_gt_poses) = splits[0]
        (val_cam_ids, val_ap_2d_poses, val_pred_poses, val_pred_rots, val_tr_poses, val_gt_poses) = splits[1]

        if config.test_split != 0:
            (test_cam_ids, test_ap_2d_poses, test_pred_poses, test_pred_rots, test_tr_poses, test_gt_poses) = self.reformat_test_data(config, splits[2])
        else:
            (test_cam_ids, test_ap_2d_poses, test_pred_poses, test_pred_rots, test_tr_poses, test_gt_poses) = self.reformat_test_data(config, 
                                                                                                                    (test_data_cam_ids, test_data_ap_2d_poses, 
                                                                                                                     test_data_pred_poses, test_data_pred_rots, 
                                                                                                                     test_data_tr_poses, test_data_gt_poses))

        self.train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_cam_ids, train_ap_2d_poses, train_pred_poses, train_pred_rots, train_tr_poses, train_gt_poses),
            batch_size=config.batch_size, shuffle=True)   
        self.val_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(val_cam_ids, val_ap_2d_poses, val_pred_poses, val_pred_rots, val_tr_poses, val_gt_poses),
            batch_size=config.batch_size, shuffle=False) 
        self.test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(test_cam_ids, test_ap_2d_poses, test_pred_poses, test_pred_rots, test_tr_poses, test_gt_poses),
            batch_size=config.eval_batch_size, shuffle=False)
        
    def reformat_test_data(self, config, test_split):
        '''
        we have to reshape test data to be (num_samples, num_cams, ...)
        '''
        (test_cam_ids, test_data_ap_2d_poses, test_pred_poses, test_pred_rots, test_tr_poses, test_gt_poses) = test_split
        # print("OG test_cam_ids: {}, test_pred_poses: {}, test_pred_rots: {}, test_tr_poses: {}, test_gt_poses: {}".format(test_cam_ids.shape, test_pred_poses.shape, 
        #                                                                                                                test_pred_rots.shape, test_tr_poses.shape, test_gt_poses.shape))
        # print(test_cam_ids[:3])
        # TODO: MAKE THIS MORE FLEXIBLE???
        test_cam_ids = test_cam_ids.view(-1, config.num_cams)
        test_data_ap_2d_poses = test_data_ap_2d_poses.view(-1, config.num_cams, config.num_kpts*3)
        test_pred_poses = test_pred_poses.view(-1, config.num_cams, config.num_kpts*3)
        test_pred_rots = test_pred_rots.view(-1, config.num_cams, 3, 3)
        test_tr_poses = test_tr_poses.view(-1, config.num_cams, config.num_kpts, 3)
        test_gt_poses = test_gt_poses.view(-1, config.num_cams, config.num_kpts, 3)
        # print("RE test_cam_ids: {}, test_pred_poses: {}, test_pred_rots: {}, test_tr_poses: {}, test_gt_poses: {}".format(test_cam_ids.shape, test_pred_poses.shape, 
        #                                                                                                                test_pred_rots.shape, test_tr_poses.shape, test_gt_poses.shape))
        # print(test_cam_ids[:3])
        return (test_cam_ids, test_data_ap_2d_poses, test_pred_poses, test_pred_rots, test_tr_poses, test_gt_poses)
    
    
    def load_data(self, config, split):
        '''
        Loads data from numpy files for the correct dataset split and converts to torch tensors
        '''
        file_prefix = config.uncertnet_data_path + config.uncertnet_file_pref + split
        # Load data from numpy files
        cam_ids = np.load(file_prefix + config.cam_ids_file)
        ap_2d_poses = np.load(file_prefix + config.ap_pred_poses_2d_file)
        pred_poses = np.load(file_prefix + config.pred_poses_3d_file)
        pred_rots = np.load(file_prefix + config.pred_camrots_file)
        tr_poses = np.load(file_prefix + config.triang_poses_3d_file)
        gt_poses = np.load(file_prefix + config.gt_poses_3d_file)

        # Only keep data for cams being used
        cam_mask = np.isin(cam_ids, config.cams)
        cam_ids = cam_ids[cam_mask]
        ap_2d_poses = ap_2d_poses[cam_mask]
        pred_poses = pred_poses[cam_mask]
        pred_rots = pred_rots[cam_mask]
        tr_poses = tr_poses[cam_mask]
        gt_poses = gt_poses[cam_mask]

        # Convert to torch tensors
        cam_ids = torch.from_numpy(cam_ids).float().to(config.device).unsqueeze(1)
        ap_2d_poses = torch.from_numpy(ap_2d_poses).float().to(config.device)
        pred_poses = torch.from_numpy(pred_poses).float().to(config.device)
        pred_rots = torch.from_numpy(pred_rots).float().to(config.device)
        tr_poses = torch.from_numpy(tr_poses).float().to(config.device)
        gt_poses = torch.from_numpy(gt_poses).float().to(config.device)

        # TEST IF CAN OVERFIT
        if (split == 'train') and (config.overfit_datalim is not None):
            print("DEBUG: Limiting training set to {} samples for overfitting".format(config.overfit_datalim), end=" ")
            cam_ids = cam_ids[:config.overfit_datalim]
            ap_2d_poses = ap_2d_poses[:config.overfit_datalim]
            pred_poses = pred_poses[:config.overfit_datalim]
            pred_rots = pred_rots[:config.overfit_datalim]
            tr_poses = tr_poses[:config.overfit_datalim]
            gt_poses = gt_poses[:config.overfit_datalim]
        return cam_ids, ap_2d_poses, pred_poses, pred_rots, tr_poses, gt_poses

    def get_splits(self, config, data_cam_ids, data_ap_2d_poses, data_pred_poses, data_pred_rots, data_tr_poses, data_gt_poses):
        val_size = int(config.val_split * len(data_cam_ids))
        test_size = int(config.test_split * len(data_cam_ids))

        # Make sure we have a multiple of config.num_cams since we want to sample a group of cameras for each test sample
        assert test_size % len(config.cams) == 0, "Test size must be a multiple of num_cams!"

        if config.test_split != 0:
            (val_cam_ids, test_cam_ids, train_cam_ids) = (data_cam_ids[:val_size], data_cam_ids[val_size:val_size + test_size], data_cam_ids[test_size:])
            (val_ap_2d_poses, test_ap_2d_poses, train_ap_2d_poses) = (data_ap_2d_poses[:val_size], data_ap_2d_poses[val_size:val_size + test_size], data_ap_2d_poses[test_size:])
            (val_pred_poses, test_pred_poses, train_pred_poses) = (data_pred_poses[:val_size], data_pred_poses[val_size:val_size + test_size], data_pred_poses[test_size:])
            (val_pred_rots, test_pred_rots, train_pred_rots) = (data_pred_rots[:val_size], data_pred_rots[val_size:val_size + test_size], data_pred_rots[test_size:])
            (val_tr_poses, test_tr_poses, train_tr_poses) = (data_tr_poses[:val_size], data_tr_poses[val_size:val_size + test_size], data_tr_poses[test_size:])
            (val_gt_poses, test_gt_poses, train_gt_poses) = (data_gt_poses[:val_size], data_gt_poses[val_size:val_size + test_size], data_gt_poses[test_size:])

            trains = [train_cam_ids, train_ap_2d_poses, train_pred_poses, train_pred_rots, train_tr_poses, train_gt_poses]
            vals = [val_cam_ids, val_ap_2d_poses, val_pred_poses, val_pred_rots, val_tr_poses, val_gt_poses]
            tests = [test_cam_ids, test_ap_2d_poses, test_pred_poses, test_pred_rots, test_tr_poses, test_gt_poses]
            return trains, vals, tests
        else:
            (val_cam_ids, train_cam_ids) = (data_cam_ids[:val_size], data_cam_ids[val_size:])
            (val_ap_2d_poses, train_ap_2d_poses) = (data_ap_2d_poses[:val_size], data_ap_2d_poses[val_size:])
            (val_pred_poses, train_pred_poses) = (data_pred_poses[:val_size], data_pred_poses[val_size:])
            (val_pred_rots, train_pred_rots) = (data_pred_rots[:val_size], data_pred_rots[val_size:])
            (val_tr_poses, train_tr_poses) = (data_tr_poses[:val_size], data_tr_poses[val_size:])
            (val_gt_poses, train_gt_poses) = (data_gt_poses[:val_size], data_gt_poses[val_size:])

            trains = [train_cam_ids, train_ap_2d_poses, train_pred_poses, train_pred_rots, train_tr_poses, train_gt_poses]
            vals = [val_cam_ids, val_ap_2d_poses, val_pred_poses, val_pred_rots, val_tr_poses, val_gt_poses]
            return trains, vals

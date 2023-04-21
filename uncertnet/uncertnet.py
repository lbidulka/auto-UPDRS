import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from uncertnet import dataset, MVP_3D
import pytorch3d.transforms as transform

# Error prediction Network
class uncert_net(torch.nn.Module):
    '''
    Network to predict the err between the lifter predicted poses and the triangulated pseudo-gt poses.
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config

        in_dim = config.num_kpts*3
        if config.use_confs:
            in_dim += config.num_kpts
        if config.use_camID:
            in_dim += 1

        self.lin = nn.Linear(in_dim, config.out_dim)

        self.dr1 = nn.Dropout(0.2)

        self.l1 = nn.Linear(in_dim, config.hidden_dim)  
        self.bn1 = nn.BatchNorm1d(config.hidden_dim)
        self.out = nn.Linear(config.hidden_dim, config.out_dim)

    def forward(self, x):
        if self.config.simple_linear:
            return self.lin(x) 
        x = self.dr1(torch.nn.functional.leaky_relu(self.bn1(self.l1(x))))
        return self.out(x)


# Wrapper for training and such
class uncert_net_wrapper():
    '''
    Wrapper around the uncertainty network to handle training, validation, and testing.
    '''
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.net = uncert_net(config).to(config.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=config.lr)
        if config.use_step_lr: 
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config.step_lr_size, gamma=0.25)
        self.criterion = torch.nn.MSELoss()
        self.dataset = dataset.H36M(config)

        # Setup pretrained 3D lifter
        self.backbone_3d = MVP_3D.Lifter().to(config.device)
        self.backbone_3d.load_state_dict(torch.load(config.lifter_ckpt_path))
        self.backbone_3d.eval()

    def train(self):
        if not self.config.uncertnet_save_ckpts: print("NOTE: Not saving checkpoints!\n")
        if self.config.use_gt_targets: print("NOTE: Using GT 3D poses for target error!\n")
        best_val_loss = 1e10
        for epoch in range(self.config.epochs):
            print("Ep: {}".format(epoch))
            train_losses, val_losses = [], []
            train_n_mpjpes, train_p_mpjpes = [], []
            val_n_mpjpes, val_p_mpjpes = [], []
            vanilla_train_n_mpjpes, vanilla_train_p_mpjpes = [], []
            vanilla_val_n_mpjpes, vanilla_val_p_mpjpes = [], []
            # Train
            for batch_idx, data in enumerate(self.dataset.train_loader):
                (cam_ids, ap_2d_poses, pred_poses, pred_rots, tr_poses, gt_poses) = data
                # Get 3D lifter preds
                # TODO: MOVE THIS TO MVP_3D WRAPPER
                kpts_2d, kpts_confs_2d = ap_2d_poses[:, :-15], ap_2d_poses[:, -15:]
                lifter_3d_preds, lifter_angle_preds = self.backbone_3d(kpts_2d, kpts_confs_2d)
                lifter_rot_preds = transform.euler_angles_to_matrix(lifter_angle_preds, convention=['X','Y','Z'])

                data = (cam_ids, ap_2d_poses, lifter_3d_preds, lifter_rot_preds, tr_poses, gt_poses)
                (cam_ids, ap_2d_poses, pred_poses, pred_rots, tr_poses, gt_poses) = data
                # Model preds and loss
                x = self.format_input(data)
                pred = self.net(x)
                train_loss = self.loss(data, pred)
                # P1 and P2 Metrics
                vanilla_poses = pred_rots.matmul(pred_poses.view(-1, 3, self.config.num_kpts))
                vanilla_poses = torch.transpose(vanilla_poses, 2, 1)
                vanilla_train_n_mpjpes.append(self.n_mpjpe(vanilla_poses.detach(), gt_poses).unsqueeze(0))
                vanilla_train_p_mpjpes.append([self.p_mpjpe(vanilla_poses.detach().cpu().numpy(), gt_poses.cpu().numpy())])

                adjusted_poses = self.get_adjusted_poses(pred, pred_poses, pred_rots, train=True)
                train_n_mpjpes.append(self.n_mpjpe(adjusted_poses.detach(), gt_poses).unsqueeze(0))
                train_p_mpjpes.append([self.p_mpjpe(adjusted_poses.detach().cpu().numpy(), gt_poses.cpu().numpy())])
                # Backprop
                train_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                # Logging
                train_losses.append(train_loss.item())
                if (batch_idx % self.config.b_print_freq == 0):
                    print(f" B {batch_idx} loss: {train_losses[-1]:.5f}")
            # Val
            with torch.no_grad():
                for batch_idx, data in enumerate(self.dataset.val_loader):
                    (cam_ids, ap_2d_poses, pred_poses, pred_rots, tr_poses, gt_poses) = data
                    # Get 3D lifter preds
                    kpts_2d, kpts_confs_2d = ap_2d_poses[:, :-15], ap_2d_poses[:, -15:]
                    lifter_3d_preds, lifter_angle_preds = self.backbone_3d(kpts_2d, kpts_confs_2d)
                    lifter_rot_preds = transform.euler_angles_to_matrix(lifter_angle_preds, convention=['X','Y','Z'])

                    data = (cam_ids, ap_2d_poses, lifter_3d_preds, lifter_rot_preds, tr_poses, gt_poses)
                    (cam_ids, ap_2d_poses, pred_poses, pred_rots, tr_poses, gt_poses) = data
                    # Uncertnet preds and loss
                    x = self.format_input(data)
                    pred = self.net(x)
                    val_losses.append(self.loss(data, pred))
                    # P1 & P2 Metrics
                    vanilla_poses = pred_rots.matmul(pred_poses.view(-1, 3, self.config.num_kpts))
                    vanilla_poses = torch.transpose(vanilla_poses, 2, 1)
                    vanilla_val_n_mpjpes.append(self.n_mpjpe(vanilla_poses, gt_poses).unsqueeze(0))
                    vanilla_val_p_mpjpes.append([self.p_mpjpe(vanilla_poses.cpu().numpy(), gt_poses.cpu().numpy())])

                    adjusted_poses = self.get_adjusted_poses(pred, pred_poses, pred_rots, train=True)
                    val_n_mpjpes.append(self.n_mpjpe(adjusted_poses, gt_poses).unsqueeze(0))
                    val_p_mpjpes.append([self.p_mpjpe(adjusted_poses.cpu().numpy(), gt_poses.cpu().numpy())])

            if self.config.use_step_lr: self.scheduler.step()

            mean_train_loss, mean_val_loss = sum(train_losses)/len(train_losses), sum(val_losses)/len(val_losses)
            train_n_mpjpe, val_n_mpjpe = torch.cat(train_n_mpjpes).mean() * 1000, torch.cat(val_n_mpjpes).mean() * 1000
            train_p_mpjpe, val_p_mpjpe = np.concatenate(train_p_mpjpes).mean() * 1000, np.concatenate(val_p_mpjpes).mean() * 1000
            vanilla_train_n_mpjpe, vanilla_val_n_mpjpe = torch.cat(vanilla_train_n_mpjpes).mean() * 1000, torch.cat(vanilla_val_n_mpjpes).mean() * 1000
            vanilla_val_p_mpjpe, vanilla_train_p_mpjpe = np.concatenate(vanilla_val_p_mpjpes).mean() * 1000, np.concatenate(vanilla_train_p_mpjpes).mean() * 1000
            
            # Logging
            if self.config.log: self.logger.log({"t_loss": mean_train_loss, "v_loss": mean_val_loss})
            if (epoch % self.config.e_print_freq == 0) or (epoch == self.config.epochs - 1):
                print(f"| mean train_loss: {mean_train_loss:.5f}, mean val_loss: {mean_val_loss:.5f}")
                print("|| 1-view n_mpjpe (P1): train: {:.3f}, val: {:.3f} | vanilla train: {:.3f}, val: {:.3f}".format(train_n_mpjpe, val_n_mpjpe, vanilla_train_n_mpjpe, vanilla_val_n_mpjpe))
                print("|| 1-view p_mpjpe (P2): train: {:.3f}, val: {:.3f} | vanilla train: {:.3f}, val: {:.3f}".format(train_p_mpjpe, val_p_mpjpe, vanilla_train_p_mpjpe, vanilla_val_p_mpjpe))
            # Save model if best val_loss
            if self.config.uncertnet_save_ckpts and (mean_val_loss < best_val_loss):
                print("Saving model (best val_loss)")
                best_val_loss = mean_val_loss
                torch.save(self.net.state_dict(), self.config.uncertnet_ckpt_path)

    def evaluate(self):
        '''
        Use the trained model to make weighted recombination predictions and compare against the pseudo-gt triangulated poses.
        '''
        # Load up the model
        print("Loading model from {}".format(self.config.uncertnet_ckpt_path))
        self.net.load_state_dict(torch.load(self.config.uncertnet_ckpt_path))
        self.net.eval()
        print("Evaluating model on test data...")
        # Go through test data
        with torch.no_grad():
            vanilla_n_mpjpes, adjusted_n_mpjpes, adjusted_sv_n_mpjpes, triangulated_n_mpjpes, naive_n_mpjpes = [], [], [], [], []
            vanilla_p_mpjpes, adjusted_p_mpjpes, adjusted_sv_p_mpjpes, triangulated_p_mpjpes, naive_p_mpjpes = [], [], [], [], []
            test_losses = []
            for batch_idx, data in enumerate(tqdm(self.dataset.test_loader)):
                # Here, data is grouped by frame, so we can get all the cam data for a frame at once
                (cam_ids, ap_2d_poses, pred_poses, pred_rots, tr_poses, gt_poses) = data
                # Get 3D lifter preds
                kpts_2d, kpts_confs_2d = ap_2d_poses[:, :-15], ap_2d_poses[:, -15:]
                lifter_3d_preds, lifter_angle_preds = self.backbone_3d(kpts_2d, kpts_confs_2d)
                lifter_rot_preds = transform.euler_angles_to_matrix(lifter_angle_preds, convention=['X','Y','Z'])

                data = (cam_ids, ap_2d_poses, lifter_3d_preds, lifter_rot_preds, tr_poses, gt_poses)
                (cam_ids, ap_2d_poses, pred_poses, pred_rots, tr_poses, gt_poses) = data
                pred_rots = pred_rots.view(-1, pred_rots.shape[-2], pred_rots.shape[-1])
                # Get uncertnet preds and use them to adjust the 3D estimator predicted poses
                x = self.format_input(data)
                pred = self.net(x)
                adjusted_poses = self.get_adjusted_poses(pred, pred_poses, pred_rots)
                adjusted_poses_singleview = self.get_adjusted_poses(pred, pred_poses, pred_rots, avg=False)
                # Loss
                test_losses.append(self.loss(data, pred, train=False))
                # vanilla method preds, rotated to each cam coords
                rot_lifter_poses = pred_rots.matmul(pred_poses.view(-1, 3, self.config.num_kpts))
                # Naive view-averaging baseline
                naive_poses = self.naive_baseline(data)
                naive_poses = pred_rots.matmul(naive_poses.view(-1, 3, self.config.num_kpts))
                # Reshape for proper err calculations
                naive_poses = naive_poses.transpose(-2, -1).reshape(-1, self.config.num_cams, 
                                                                 naive_poses.shape[-1], naive_poses.shape[-2])
                rot_lifter_poses = rot_lifter_poses.transpose(-2, -1).reshape(-1, self.config.num_cams, 
                                                                 rot_lifter_poses.shape[-1], rot_lifter_poses.shape[-2])
                rot_lifter_poses = rot_lifter_poses.reshape(-1, rot_lifter_poses.shape[-2], rot_lifter_poses.shape[-1])
                adjusted_poses = adjusted_poses.reshape(-1, adjusted_poses.shape[-2], adjusted_poses.shape[-1])
                adjusted_poses_singleview = adjusted_poses_singleview.reshape(-1, adjusted_poses_singleview.shape[-2], adjusted_poses_singleview.shape[-1])
                naive_poses = naive_poses.reshape(-1, naive_poses.shape[-2], naive_poses.shape[-1])
                gt_poses = gt_poses.reshape(-1, gt_poses.shape[-2], gt_poses.shape[-1])
                tr_poses = tr_poses.reshape(-1, tr_poses.shape[-2], tr_poses.shape[-1])
                # N-MPJPE (P1)
                vanilla_n_mpjpes.append(self.n_mpjpe(rot_lifter_poses, gt_poses).unsqueeze(0))
                adjusted_n_mpjpes.append(self.n_mpjpe(adjusted_poses, gt_poses).unsqueeze(0))
                adjusted_sv_n_mpjpes.append(self.n_mpjpe(adjusted_poses_singleview, gt_poses).unsqueeze(0))
                naive_n_mpjpes.append(self.n_mpjpe(naive_poses, gt_poses).unsqueeze(0))
                triangulated_n_mpjpes.append(self.n_mpjpe(tr_poses, gt_poses).unsqueeze(0))
                # P-MPJPE (P2)
                vanilla_p_mpjpes.append([self.p_mpjpe(rot_lifter_poses.cpu().numpy(), gt_poses.cpu().numpy())])
                adjusted_p_mpjpes.append([self.p_mpjpe(adjusted_poses.cpu().numpy(), gt_poses.cpu().numpy())])
                adjusted_sv_p_mpjpes.append([self.p_mpjpe(adjusted_poses_singleview.cpu().numpy(), gt_poses.cpu().numpy())])
                naive_p_mpjpes.append([self.p_mpjpe(naive_poses.cpu().numpy(), gt_poses.cpu().numpy())])
                triangulated_p_mpjpes.append([self.p_mpjpe(tr_poses.cpu().numpy(), gt_poses.cpu().numpy())])
            
            # Logging
            mean_test_loss = sum(test_losses)/len(test_losses)

            vanilla_n_mpjpe = torch.cat(vanilla_n_mpjpes).mean() * 1000
            adjusted_n_mpjpe = torch.cat(adjusted_n_mpjpes).mean() * 1000
            adjusted_sv_n_mpjpe = torch.cat(adjusted_sv_n_mpjpes).mean() * 1000
            triangulated_n_mpjpe = torch.cat(triangulated_n_mpjpes).mean() * 1000
            naive_n_mpjpe = torch.cat(naive_n_mpjpes).mean() * 1000

            vanilla_p_mpjpe = np.concatenate(vanilla_p_mpjpes).mean() * 1000
            adjusted_p_mpjpe = np.concatenate(adjusted_p_mpjpes).mean() * 1000
            adjusted_sv_p_mpjpe = np.concatenate(adjusted_sv_p_mpjpes).mean() * 1000
            triangulated_p_mpjpe = np.concatenate(triangulated_p_mpjpes).mean() * 1000
            naive_p_mpjpe = np.concatenate(naive_p_mpjpes).mean() * 1000

            print("\nTest loss: {:.3f}".format(mean_test_loss))

            print("n_mpjpe (P1):")
            print("SV    | adjusted : {:.3f}, Vanilla: {:.3f}, triangulated: {:.3f}".format(adjusted_sv_n_mpjpe, vanilla_n_mpjpe, triangulated_n_mpjpe))
            print("avg'd | adjusted : {:.3f}, Vanilla: {:.3f}".format(adjusted_n_mpjpe, naive_n_mpjpe))

            print("\np_mpjpe (P2):")
            print("SV    | adjusted : {:.3f}, Vanilla: {:.3f}, triangulated: {:.3f}".format(adjusted_sv_p_mpjpe, vanilla_p_mpjpe, triangulated_p_mpjpe))
            print("avg'd | adjusted : {:.3f}, Vanilla: {:.3f}".format(adjusted_p_mpjpe, naive_p_mpjpe))
            
            # print("n_mpjpe (P1): adjusted SV: {:.3f}, adjusted avg'd: {:.3f} | Vanilla avg'd: {:.3f}, Vanilla SV: {:.3f}, triangulated: {:.3f}".format(adjusted_sv_n_mpjpe, adjusted_n_mpjpe, 
            #                                                                    naive_n_mpjpe, vanilla_n_mpjpe, triangulated_n_mpjpe))
            # print("p_mpjpe (P2): adjusted SV: {:.3f}, adjusted avg'd: {:.3f} | Vanilla avg'd: {:.3f}, Vanilla SV: {:.3f}, triangulated: {:.3f}".format(adjusted_sv_p_mpjpe, adjusted_p_mpjpe, 
            #                                                                    naive_p_mpjpe, vanilla_p_mpjpe, triangulated_p_mpjpe))
            if self.config.log: self.logger.log({"Vanilla_n_mpjpe": vanilla_n_mpjpe, "Vanilla_p_mpjpe": vanilla_p_mpjpe, 
                                                 "UncertNet_sv_n_mpjpe": adjusted_sv_n_mpjpe, "UncertNet_sv_p_mpjpe": adjusted_sv_p_mpjpe,
                                                 "Adjusted_n_mpjpe": adjusted_n_mpjpe, "Adjusted_p_mpjpe": adjusted_p_mpjpe, 
                                                 "Triangulated_n_mpjpe": triangulated_n_mpjpe, "Triangulated_p_mpjpe": triangulated_p_mpjpe,
                                                 "Naive_n_mpjpe": naive_n_mpjpe})

    def loss(self, data, pred_err, train=True, compute_criterion=True):
        '''
        - Reshape inputs and compute the per-sample mean per joint position error in mm.
        - Compute loss between model pred and this error
        '''
        (cam_ids, ap_2d_poses, pred_poses, pred_rots, tr_poses, gt_poses) = data
        pred_poses = pred_poses.view(-1, 3, self.config.num_kpts)
        
        # If we are evaluating, we have to reshape the data to be per sample instead of grouped in cam batches
        if not train:
            pred_rots = pred_rots.view(-1, 3, 3)
            tr_poses = tr_poses.view(-1, self.config.num_kpts, 3)
            gt_poses = gt_poses.view(-1, self.config.num_kpts, 3)

        rot_poses = pred_rots.matmul(pred_poses)
        comparison_poses = gt_poses if self.config.use_gt_targets else tr_poses

        err = torch.norm(rot_poses.transpose(2, 1) - comparison_poses, dim=len(comparison_poses.shape) - 1).mean(dim=1, keepdim=True)
        if self.config.out_per_kpt:
            diff = rot_poses.transpose(2, 1) - comparison_poses
            err = torch.norm(diff, dim=2)
        if self.config.out_directional:
            pred_err = pred_err.view(-1, self.config.num_kpts, 3)                
            err = rot_poses.transpose(2, 1) - comparison_poses
        err *= self.config.err_scale

        # Compute loss
        if compute_criterion: err = self.criterion(pred_err, err)
        return err
    
    def get_adjusted_poses(self, pred_err, pred_poses, pred_rots, train=False, avg=True):
        '''
        Adjust the backbone 3D kpt predictions, and rotate them to each cam coords
        '''
        rot_pred_poses = pred_rots.matmul(pred_poses.view(-1, 3, self.config.num_kpts))
        if self.config.out_directional:
            # Reshape 
            if not train: 
                pred_err = pred_err.view(-1, self.config.num_cams, self.config.num_kpts, 3)
                rot_pred_poses = rot_pred_poses.view(-1, self.config.num_cams, 3, self.config.num_kpts)
            else:
                pred_err = pred_err.view(-1, self.config.num_kpts, 3)
            # Get corrected poses
            if not self.config.out_per_kpt:
                raise ValueError("Directional output must be per kpt")
            # Get error adjusted poses
            corrected_poses = rot_pred_poses.transpose(-2, -1) - (pred_err / self.config.err_scale)
            if not train: 
                # Rotate all corrected poses to canonical pose
                corrected_poses = corrected_poses.reshape(-1, corrected_poses.shape[-2], corrected_poses.shape[-1]).transpose(-2, -1)
                corrected_poses = torch.inverse(pred_rots).matmul(corrected_poses).reshape(-1, self.config.num_cams, 3, self.config.num_kpts)
                # Avg the corrected poses and repeat for each cam if desired
                if avg:
                    adjusted_poses = torch.mean(corrected_poses, dim=1).unsqueeze(1)
                    adjusted_poses = adjusted_poses.repeat(1, self.config.num_cams, 1, 1)
                else:
                    adjusted_poses = corrected_poses
                # Rotate back to cam coords
                adjusted_poses = pred_rots.matmul(adjusted_poses.reshape(-1, adjusted_poses.shape[-2], adjusted_poses.shape[-1]))
                adjusted_poses = adjusted_poses.reshape(-1, self.config.num_cams, adjusted_poses.shape[-2], adjusted_poses.shape[-1]).transpose(-2, -1)
                return adjusted_poses
            else:
                return corrected_poses
        else:
            weights = self.create_weights(pred_err)
            adjusted_poses = self.reweight_poses(rot_pred_poses, weights)
            return adjusted_poses
    
    def reweight_poses(self, rot_pred_poses, weights):
        '''
        Combine the predicted poses using the weights
        '''
        rot_pred_poses = rot_pred_poses.view(-1, self.config.num_cams, 3, self.config.num_kpts)
        if self.config.out_per_kpt:
            reweighted_poses = torch.sum(rot_pred_poses * weights.unsqueeze(-2), dim=1).view(-1, 1, 3, self.config.num_kpts)
        else: 
            reweighted_poses = torch.sum(rot_pred_poses * weights.unsqueeze(-1), dim=1).view(-1, 1, 3, self.config.num_kpts)
        # Repeat reweighted poses to all cams
        reweighted_poses = reweighted_poses.repeat(1, self.config.num_cams, 1, 1)
        return reweighted_poses

    def create_weights(self, pred):
        '''
        Create weights for recombining outputs from the model output
        '''
        if self.config.out_per_kpt:
            pred = pred.view(-1, self.config.num_cams, self.config.num_kpts)
        else:
            pred = pred.view(-1, self.config.num_cams)

        # weights = torch.softmax(-pred, dim=1)
        # pred = torch.abs(pred)  # TODO: IS THIS SKETCH?
        pred_min = torch.min(pred, dim=1, keepdim=True)[0]
        pred_max = torch.max(pred, dim=1, keepdim=True)[0]
        pred_adj = (pred - pred_min) / (pred_max - pred_min)    # Good range one
        weights = torch.softmax(-pred_adj, dim=1)
        return weights

    def naive_baseline(self, data):
        '''
        Get a naive equal avg of the 3D backbone predicted poses
        '''
        (cam_ids, ap_2d_poses, pred_poses, pred_rots, tr_poses, gt_poses) = data
        pred_rots = pred_rots.view(-1, pred_rots.shape[2], pred_rots.shape[3])
        # Make equal weights and combine
        weights = torch.ones((cam_ids.shape[0], self.config.num_cams)).to(self.config.device) / self.config.num_cams
        naive_poses = torch.sum(pred_poses * weights.unsqueeze(-1), dim=1).view(-1, 1, 3, self.config.num_kpts)
        # Repeat reweighted poses to all cams and rotate each to corresponding cam coords
        naive_poses = naive_poses.repeat(1, self.config.num_cams, 1, 1)
        return naive_poses

    def format_input(self, data):
        '''
        Create correct input data for the model, depending on the config
        '''
        (cam_ids, ap_2d_poses, pred_poses, pred_rots, tr_poses, gt_poses) = data
        if len(ap_2d_poses.shape) == 2:
            ap_2d_confs = ap_2d_poses[:, -15:]
        elif len(ap_2d_poses.shape) == 3:
            ap_2d_confs = ap_2d_poses[:, :, -15:]
        else:
            raise ValueError("ap_2d_poses shape is not 2 or 3?")

        # We want input data scale to be mm, not m
        x = pred_poses.view(-1, self.config.num_kpts*3) #* self.config.err_scale # TODO: MAKE THIS MORE FLEXIBLE, DEPENDANT ON 3D BACKBONE
        # Add inputs as per config
        if self.config.use_confs:
            x = torch.cat((x, ap_2d_confs.view(-1, self.config.num_kpts)), dim=1)
        if self.config.use_camID:
            x = torch.cat((x, cam_ids.view(-1, 1)), dim=1)

        return x

    def mpjpe(self, predicted, target, mean=True):
        """
        Mean per-joint position error (i.e. mean Euclidean distance),
        often referred to as "Protocol #1" in many papers.
        """
        assert predicted.shape == target.shape
        if mean:
            err = torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))
        else:
            err = torch.norm(predicted - target, dim=len(target.shape)-1)
        return err

    def n_mpjpe(self, predicted, target, mean=True):
        """
        Normalized MPJPE (scale only), adapted from:
        https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
        """
        assert predicted.shape == target.shape
        
        norm_predicted = torch.sum(predicted**2, dim=-1, keepdim=True)
        norm_target = torch.sum(target*predicted, dim=-1, keepdim=True)
        if mean:
            norm_predicted = torch.mean(norm_predicted, dim=-2, keepdim=True)
            norm_target = torch.mean(norm_target, dim=-2, keepdim=True)
        scale = norm_target / (norm_predicted+0.0001)
        return self.mpjpe(scale * predicted, target, mean=mean)

    def p_mpjpe(self, predicted, target):
        """
        Pose error: MPJPE after rigid alignment (scale, rotation, and translation),
        often referred to as "Protocol #2" in many papers.
        """
        assert predicted.shape == target.shape
        
        muX = np.mean(target, axis=1, keepdims=True)
        muY = np.mean(predicted, axis=1, keepdims=True)
        
        X0 = target - muX
        Y0 = predicted - muY

        normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
        normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
        
        X0 /= normX
        Y0 /= normY

        H = np.matmul(X0.transpose(0, 2, 1), Y0)
        U, s, Vt = np.linalg.svd(H)
        V = Vt.transpose(0, 2, 1)
        R = np.matmul(V, U.transpose(0, 2, 1))

        # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
        sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
        V[:, :, -1] *= sign_detR
        s[:, -1] *= sign_detR.flatten()
        R = np.matmul(V, U.transpose(0, 2, 1)) # Rotation

        tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)

        a = tr * normX / normY # Scale
        t = muX - a*np.matmul(muY, R) # Translation
        
        # Perform rigid transformation on the input
        predicted_aligned = a*np.matmul(predicted, R) + t
        
        # Return MPJPE
        return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1))


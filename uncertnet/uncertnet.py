import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from uncertnet import dataset

# Network
class uncert_net(torch.nn.Module):
    '''
    Network to predict the err between the lifter predicted poses and the triangulated pseudo-gt poses.
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        # TODO: 2 branches, one for poses, one for kpts and one for cam_id. Make branches embed to same size then combine, put through some end layers and output
        # TODO: Incorporate confidences
        # TODO: Think about the relative size of the cam_id (1) and the kpts (15*4)...

        in_dim = config.num_kpts*3
        if config.use_confs:
            in_dim += config.num_kpts
        if config.use_camID:
            in_dim += 1

        self.l1 = nn.Linear(in_dim, config.hidden_dim)  
        self.bn1 = nn.BatchNorm1d(config.hidden_dim)
        self.l2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.bn2 = nn.BatchNorm1d(config.hidden_dim)
        self.l3 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.bn3 = nn.BatchNorm1d(config.hidden_dim)
        # self.l4 = nn.Linear(config.hidden_dim, config.hidden_dim)
        # self.bn4 = nn.BatchNorm1d(config.hidden_dim)
        self.out = nn.Linear(config.hidden_dim, config.out_dim)

    def forward(self, x):
        # x: [num_kpts * (x,y,z), cam_id]
        x = torch.relu(self.bn3(self.l1(x)))
        x = torch.relu(self.bn3(self.l2(x)))
        x = torch.relu(self.bn3(self.l3(x)))
        # x = torch.relu(self.bn3(self.l4(x)))
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
        self.criterion = torch.nn.MSELoss()
        self.dataset = dataset.H36M(config)

    def train(self):
        if not self.config.uncertnet_save_ckpts: print("NOTE: Not saving checkpoints!\n")
        best_val_loss = 1e10
        for epoch in range(self.config.epochs):
            print("Ep: {}".format(epoch))
            train_losses = []
            val_losses = []
            # Train
            for batch_idx, data in enumerate(self.dataset.train_loader):
                (cam_ids, ap_2d_poses, pred_poses, pred_rots, tr_poses, gt_poses) = data
                x = self.format_input(data)
                pred = self.net(x)
                lifter_err = self.mpjpe_mm(data)
                train_loss = self.criterion(pred, lifter_err)
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
                    x = self.format_input(data)
                    pred = self.net(x)
                    lifter_err = self.mpjpe_mm(data)
                    val_loss = self.criterion(pred, lifter_err) 
                    # Logging
                    val_losses.append(val_loss)

            mean_train_loss, mean_val_loss = sum(train_losses)/len(train_losses), sum(val_losses)/len(val_losses)
            # Save model if best val_loss
            if self.config.uncertnet_save_ckpts and (mean_val_loss < best_val_loss):
                print("Saving model (mean val_loss: {:.5f})".format(mean_val_loss))
                best_val_loss = mean_val_loss
                torch.save(self.net.state_dict(), self.config.uncertnet_ckpt_path)
            # Logging
            if self.config.log: self.logger.log({"t_loss": mean_train_loss, "v_loss": mean_val_loss})
            if (epoch % self.config.e_print_freq == 0) or (epoch == self.config.epochs - 1):
                print(f"|| mean train_loss: {mean_train_loss:.5f}, mean val_loss: {mean_val_loss:.5f}") #, \
                    #   mean tri_gt_mpjpe_err: {epoch_tri_gt_mpjpe_err:.6f}")

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
            vanilla_n_mpjpes, reweighted_n_mpjpes, triangulated_n_mpjpes, naive_n_mpjpes = [], [], [], []
            vanilla_p_mpjpes, reweighted_p_mpjpes, triangulated_p_mpjpes, naive_p_mpjpes = [], [], [], []
            for batch_idx, data in enumerate(tqdm(self.dataset.test_loader)):
                # Here, data is grouped by frame, so we can get all the cam data for a frame at once
                (cam_ids, ap_2d_poses, pred_poses, pred_rots, tr_poses, gt_poses) = data
                pred_rots = pred_rots.view(-1, pred_rots.shape[2], pred_rots.shape[3])
                # print("cam_ids: {}, pred_poses: {}, pred_rots: {}, tr_poses: {}, gt_poses: {}".format(cam_ids.shape, pred_poses.shape, 
                #                                                                                                        pred_rots.shape, tr_poses.shape, gt_poses.shape))
                
                # Get model pred
                x = self.format_input(data)
                pred = self.net(x)

                # Make reweighted predictions, and rotate them to each cam coords
                weights = self.create_weights(pred)
                reweighted_poses = self.reweight_poses(pred_poses, weights)
                reweighted_poses = pred_rots.matmul(reweighted_poses.view(-1, 3, self.config.num_kpts))

                # Get vanilla method preds, rotated to each cam coords
                rot_lifter_poses = pred_rots.matmul(pred_poses.view(-1, 3, self.config.num_kpts))

                # Get Naive baseline
                naive_preds = self.naive_baseline(data)
                naive_preds = pred_rots.matmul(naive_preds.view(-1, 3, self.config.num_kpts))

                # Get errs btw various preds and gt poses
                gt_poses = gt_poses.view(-1, self.config.num_kpts, 3)  
                tr_poses = tr_poses.view(-1, self.config.num_kpts, 3)  
                reweighted_poses = torch.transpose(reweighted_poses, 2, 1)
                naive_poses = torch.transpose(naive_preds, 2, 1)
                rot_lifter_poses = torch.transpose(rot_lifter_poses, 2, 1)
                # print("reweighted_poses: {}, rot_lifter_poses: {}, tr_poses: {}, gt_poses: {}".format(reweighted_poses.shape, rot_lifter_poses.shape, 
                #                                                                                       tr_poses.shape, gt_poses.shape))

                # N-MPJPE (P1)
                vanilla_n_mpjpes.append(self.n_mpjpe(rot_lifter_poses, gt_poses).unsqueeze(0))
                reweighted_n_mpjpes.append(self.n_mpjpe(reweighted_poses, gt_poses).unsqueeze(0))
                naive_n_mpjpes.append(self.n_mpjpe(naive_poses, gt_poses).unsqueeze(0))
                triangulated_n_mpjpes.append(self.n_mpjpe(tr_poses, gt_poses).unsqueeze(0))
                # P-MPJPE (P2)
                vanilla_p_mpjpes.append([self.p_mpjpe(rot_lifter_poses.cpu().numpy(), gt_poses.cpu().numpy())])
                reweighted_p_mpjpes.append([self.p_mpjpe(reweighted_poses.cpu().numpy(), gt_poses.cpu().numpy())])
                naive_p_mpjpes.append([self.p_mpjpe(naive_poses.cpu().numpy(), gt_poses.cpu().numpy())])
                triangulated_p_mpjpes.append([self.p_mpjpe(tr_poses.cpu().numpy(), gt_poses.cpu().numpy())])
            
            # Logging
            vanilla_n_mpjpe = torch.cat(vanilla_n_mpjpes).mean() * 1000
            reweighted_n_mpjpe = torch.cat(reweighted_n_mpjpes).mean()  * 1000
            triangulated_n_mpjpe = torch.cat(triangulated_n_mpjpes).mean() * 1000
            naive_n_mpjpe = torch.cat(naive_n_mpjpes).mean() * 1000

            vanilla_p_mpjpe = np.concatenate(vanilla_p_mpjpes).mean() * 1000
            reweighted_p_mpjpe = np.concatenate(reweighted_p_mpjpes).mean()  * 1000
            triangulated_p_mpjpe = np.concatenate(triangulated_p_mpjpes).mean() * 1000
            naive_p_mpjpe = np.concatenate(naive_p_mpjpes).mean() * 1000

            print("\nn_mpjpe (P1): reweighted: {:.3f}, naive: {:.3f} | Vanilla: {:.3f}, triangulated: {:.3f}".format(reweighted_n_mpjpe, naive_n_mpjpe, 
                                                                                                                                      vanilla_n_mpjpe, triangulated_n_mpjpe))
            print("p_mpjpe (P2): reweighted: {:.3f}, naive: {:.3f} | Vanilla: {:.3f}, triangulated: {:.3f}".format(reweighted_p_mpjpe, naive_p_mpjpe, 
                                                                                                                                      vanilla_p_mpjpe, triangulated_p_mpjpe))
            if self.config.log: self.logger.log({"Vanilla_n_mpjpe": vanilla_n_mpjpe, 
                                                 "Reweighted_n_mpjpe": reweighted_n_mpjpe, 
                                                 "Triangulated_n_mpjpe": triangulated_n_mpjpe,
                                                 "Naive_n_mpjpe": naive_n_mpjpe})

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
        x = pred_poses.view(-1, self.config.num_kpts*3) * self.config.err_scale # TODO: MAKE THIS MORE FLEXIBLE, DEPENDANT ON 3D BACKBONE
        # Add inputs as per config
        if self.config.use_confs:
            x = torch.cat((x, ap_2d_confs.view(-1, self.config.num_kpts)), dim=1)
        if self.config.use_camID:
            x = torch.cat((x, cam_ids.view(-1, 1)), dim=1)

        return x
    
    def reweight_poses(self, pred_poses, weights):
        '''
        Combine the predicted poses using the weights
        '''
        if self.config.out_per_kpt:
            reweighted_poses = torch.sum(pred_poses, dim=1).view(-1, 1, 3, self.config.num_kpts)
        else: 
            reweighted_poses = torch.sum(pred_poses * weights.unsqueeze(-1), dim=1).view(-1, 1, 3, self.config.num_kpts)
        # Repeat reweighted poses to all cams and rotate each to corresponding cam coords
        reweighted_poses = reweighted_poses.repeat(1, self.config.num_cams, 1, 1)
        return reweighted_poses

    def create_weights(self, pred):
        '''
        Create weights from the model output
        '''
        if self.config.out_per_kpt:
            pred = pred.view(-1, self.config.num_cams, self.config.num_kpts)
            weights = torch.softmax(-pred, dim=1)
        #     pred_min = torch.min(pred, dim=1, keepdim=True)[0]
        #     pred_max = torch.max(pred, dim=1, keepdim=True)[0]
        #     # pred_adj = (pred) / (pred_max)
        #     pred_adj = pred - pred_min
        #     weights = torch.softmax(-pred_adj, dim=1)
        #     print("\npred: \n{}, \nsf: \n{}, \npred_adj: \n{}, \nsf_adj: \n{}".format(pred[0], torch.softmax(-pred, dim=1)[0], pred_adj[0], weights[0]))
        else:
            pred = pred.view(-1, self.config.num_cams)
            weights = torch.softmax(-pred, dim=1)
            # pred_min = torch.min(pred, dim=1, keepdim=True)[0]
            # pred_max = torch.max(pred, dim=1, keepdim=True)[0]
            # pred_adj = (pred) / (pred_max)
            # # print("pred: {}, sf: {}, pred_adj: {}, sf_adj: {}".format(pred, torch.softmax(-pred, dim=1), 
            # #                                                           pred_adj, torch.softmax(-pred_adj, dim=1)))
            # weights = torch.softmax(-pred_adj, dim=1)
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

    def mpjpe_mm(self, data):
        '''
        Reshape inputs and compute the per-sample mean per joint position error in mm.
        Used for Loss
        '''
        (cam_ids, ap_2d_poses, pred_poses, pred_rots, tr_poses, gt_poses) = data
        rot_poses = pred_rots.matmul(pred_poses.reshape(-1, 3, self.config.num_kpts))
        if self.config.out_per_kpt:
            err = torch.norm(rot_poses.transpose(2, 1) - tr_poses, dim=len(tr_poses.shape) - 1)
        else:
            err = torch.norm(rot_poses.transpose(2, 1) - tr_poses, dim=len(tr_poses.shape) - 1).mean(dim=1, keepdim=True)
        return err * self.config.err_scale

    def mpjpe(self, predicted, target):
        """
        Mean per-joint position error (i.e. mean Euclidean distance),
        often referred to as "Protocol #1" in many papers.
        """
        assert predicted.shape == target.shape
        return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))

    def n_mpjpe(self, predicted, target):
        """
        Normalized MPJPE (scale only), adapted from:
        https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning/blob/master/losses/poses.py
        """
        assert predicted.shape == target.shape
        
        norm_predicted = torch.mean(torch.sum(predicted**2, dim=-1, keepdim=True), dim=-2, keepdim=True)
        norm_target = torch.mean(torch.sum(target*predicted, dim=-1, keepdim=True), dim=-2, keepdim=True)
        scale = norm_target / (norm_predicted+0.0001)
        return self.mpjpe(scale * predicted, target)

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


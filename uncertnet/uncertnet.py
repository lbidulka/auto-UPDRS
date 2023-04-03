import torch
import torch.nn as nn
import torch.nn.functional as F
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

        self.l1 = nn.Linear(in_dim, 512)  
        self.bn1 = nn.BatchNorm1d(512)
        self.l2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.l3 = nn.Linear(512, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.out = nn.Linear(512, config.out_dim)

    def forward(self, x):
        # x: [num_kpts * (x,y,z), cam_id]
        x = torch.relu(self.bn1(self.l1(x)))
        x = torch.relu(self.bn2(self.l2(x)))
        x = torch.relu(self.bn3(self.l3(x)))
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
                (cam_ids, pred_poses, pred_rots, tr_poses, gt_poses) = data
                # Combine the data into one tensor and get model pred
                x = self.format_input(data)
                pred = self.net(x)
                # Get distance from lifter preds to triang preds, and compare to model pred
                rot_poses = pred_rots.matmul(pred_poses.reshape(-1, 3, self.config.num_kpts))
                if self.config.out_per_kpt:
                    raise NotImplementedError
                else:
                    triang_mpjpe_err = torch.norm(rot_poses.transpose(2, 1) - tr_poses, dim=len(tr_poses.shape) - 1).mean(dim=1, keepdim=True)
                train_loss = self.criterion(pred, triang_mpjpe_err)
                # Backprop
                self.optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                # Logging
                train_losses.append(train_loss.item())
                if (batch_idx % self.config.b_print_freq == 0):
                    print(f" B {batch_idx} loss: {train_losses[-1]:.5f}")
            # Val
            with torch.no_grad():
                for batch_idx, data in enumerate(self.dataset.val_loader):
                    (cam_ids, pred_poses, pred_rots, tr_poses, gt_poses) = data
                    # Combine the data into one tensor and get model pred
                    x = self.format_input(data)
                    pred = self.net(x)
                    # Get distance from lifter preds to triang preds, and compare to model pred
                    rot_poses = pred_rots.matmul(pred_poses.reshape(-1, 3, self.config.num_kpts))
                    if self.config.out_per_kpt:
                        raise NotImplementedError
                    else:
                        triang_mpjpe_err = torch.norm(rot_poses.transpose(2, 1) - tr_poses, dim=len(tr_poses.shape) - 1).mean(dim=1, keepdim=True)
                    val_loss = self.criterion(pred, triang_mpjpe_err)
                    val_losses.append(val_loss)

            mean_train_loss, mean_val_loss = sum(train_losses)/len(train_losses), sum(val_losses)/len(val_losses)
            # Save model if best val
            if self.config.uncertnet_save_ckpts and (mean_val_loss < best_val_loss):
                print("Saving model (mean_val_loss: {:.5f})".format(val_loss))
                best_val_loss = mean_val_loss
                torch.save(self.net.state_dict(), self.config.uncertnet_ckpt_path)
            # Logging
            if self.config.log: self.logger.log({"t_loss": mean_train_loss, "v_loss": mean_val_loss})
            if (epoch % self.config.e_print_freq == 0) or (epoch == self.config.epochs - 1):
                print(f"--> Ep{epoch} avg train_loss: {mean_train_loss:.5f}, avg val_loss: {mean_val_loss:.5f}") #, \
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
            vanilla_errs = []
            reweighted_errs = []
            triangulated_errs = []
            naive_errs = []
            for batch_idx, data in enumerate(tqdm(self.dataset.test_loader)):
                # Here, data is grouped by frame, so we can get all the cam data for a frame at once
                (cam_ids, pred_poses, pred_rots, tr_poses, gt_poses) = data
                pred_rots = pred_rots.view(-1, pred_rots.shape[2], pred_rots.shape[3])
                # print("cam_ids: {}, pred_poses: {}, pred_rots: {}, tr_poses: {}, gt_poses: {}".format(cam_ids.shape, pred_poses.shape, 
                #                                                                                                        pred_rots.shape, tr_poses.shape, gt_poses.shape))
                
                # Get model pred
                x = self.format_input(data)
                pred = self.net(x)
                pred = pred.view(-1, self.config.num_cams)

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
                reweighted_mean_err = self.n_mpjpe(reweighted_poses, gt_poses).unsqueeze(0) 
                vanilla_mean_err = self.n_mpjpe(rot_lifter_poses, gt_poses).unsqueeze(0)
                triangulated_mean_err = self.n_mpjpe(tr_poses, gt_poses).unsqueeze(0)
                naive_mean_err = self.n_mpjpe(naive_poses, gt_poses).unsqueeze(0) 

                vanilla_errs.append(vanilla_mean_err)
                reweighted_errs.append(reweighted_mean_err)
                naive_errs.append(naive_mean_err)
                triangulated_errs.append(triangulated_mean_err)
            
            # Logging
            vanilla_err = torch.cat(vanilla_errs).mean() * 1000
            reweighted_err = torch.cat(reweighted_errs).mean()  * 1000
            triangulated_err = torch.cat(triangulated_errs).mean() * 1000
            naive_err = torch.cat(naive_errs).mean() * 1000
            print("Vanilla err: {:.3f}, reweighted err: {:.3f}, triangulated err: {:.3f}, Naive_err: {:.3f}".format(vanilla_err, reweighted_err, triangulated_err, naive_err))
            if self.config.log: self.logger.log({"Vanilla_err": vanilla_err, 
                                                 "Reweighted_err": reweighted_err, 
                                                 "Triangulated_err": triangulated_err,
                                                 "Naive_err": naive_err})

    def format_input(self, data):
        '''
        Create correct input data for the model, depending on the config
        '''
        (cam_ids, pred_poses, pred_rots, tr_poses, gt_poses) = data

        if self.config.use_camID:
            x = torch.cat((cam_ids.view(-1, 1), pred_poses.view(-1, self.config.num_kpts*3),), dim=1)
        else:
            x = pred_poses.view(-1, self.config.num_kpts*3)

        return x
    
    def reweight_poses(self, pred_poses, weights):
        '''
        Combine the predicted poses using the weights
        '''
        reweighted_poses = torch.sum(pred_poses * weights.unsqueeze(-1), dim=1).view(-1, 1, 3, self.config.num_kpts)
        # Repeat reweighted poses to all cams and rotate each to corresponding cam coords
        reweighted_poses = reweighted_poses.repeat(1, self.config.num_cams, 1, 1)
        return reweighted_poses

    def create_weights(self, pred):
        '''
        Create weights from the model output
        '''
        weights = torch.softmax(-pred, dim=1)
        return weights

    def naive_baseline(self, data):
        '''
        Get a naive equal avg of the 3D backbone predicted poses
        '''
        (cam_ids, pred_poses, pred_rots, tr_poses, gt_poses) = data
        pred_rots = pred_rots.view(-1, pred_rots.shape[2], pred_rots.shape[3])
        # Make equal weights and combine
        weights = torch.ones((cam_ids.shape[0], self.config.num_cams)).to(self.config.device) / self.config.num_cams
        naive_poses = torch.sum(pred_poses * weights.unsqueeze(-1), dim=1).view(-1, 1, 3, self.config.num_kpts)
        # Repeat reweighted poses to all cams and rotate each to corresponding cam coords
        naive_poses = naive_poses.repeat(1, self.config.num_cams, 1, 1)
        return naive_poses

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



from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, confusion_matrix, classification_report, precision_recall_fscore_support

import logging
import time
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

class DeepSADTrainer(BaseTrainer):

    def __init__(self, c, lamda, radius, gamma, eta: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), ascent_step_size=0.001, ascent_num_steps: int=50, batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        # Deep SAD parameters
        self.c = torch.tensor(c, device=self.device) if c is not None else None
        self.eta = eta
        
        # adv training parameters
        self.lamda = lamda
        self.radius = radius
        self.gamma = gamma
        self.ascent_num_steps = ascent_num_steps
        self.ascent_step_size = ascent_step_size
        
        # Optimization parameters
        self.eps = 1e-6

        # Results
        self.train_time = None
        self.test_auc_cluster = None
        self.test_auc_pred = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, dataset: BaseADDataset, net: BaseNet, wotrans: bool, woadv: bool):

        logger = logging.getLogger()

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device for network
        self.net = net.to(self.device)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Initialize hypersphere center c (if c not loaded)
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, self.net)
            logger.info('Center c initialized.')

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        self.net.train()
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            epoch_loss = 0.0
            epoch_tran_loss = 0.0
            epoch_adv_loss = 0.0
            epoch_bce_loss = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, targets, semi_targets, _, _ = data # rests of inputs doesn't have labels
                known_idx = np.argwhere(np.isin(semi_targets, [-1,1])).flatten()
                inputs, targets, semi_targets = inputs.to(self.device), targets.to(self.device).to(torch.float), semi_targets.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize
                tran_outputs = self.net(inputs)
                if not wotrans:
                    dist = torch.sum((tran_outputs - self.c) ** 2, dim=1)
                    # losses = torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
                    tran_losses = dist #torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
                    tran_loss = torch.mean(tran_losses)
                    epoch_tran_loss += tran_loss.item()
                else:
                    tran_loss = torch.tensor(0)
                    epoch_tran_loss = 0
                    
                # BCE
                pre_net_outputs = self.net.pred_adv(tran_outputs)
                pre_logits = torch.squeeze(pre_net_outputs, dim = 1)
                bce_losses = F.binary_cross_entropy_with_logits(pre_logits[known_idx], targets[known_idx], reduction='none')
                bce_loss = torch.mean(bce_losses)
                epoch_bce_loss += bce_loss.item()
                ##### mask deng
                
                # AdvLoss 
                if not woadv:
                    adv_losses = self.one_class_adv_loss(inputs)
                    adv_loss = torch.mean(adv_losses)
                    epoch_adv_loss += adv_loss.item()
                else:
                    adv_loss = torch.tensor(0)
                    epoch_adv_loss = 0
                
                loss = tran_loss + (adv_loss * self.lamda + bce_loss) * self.eta
                
                if wotrans:
                    loss = (adv_loss * self.lamda + bce_loss) * self.eta
                    
                if woadv:
                    loss = tran_loss + (bce_loss) * self.eta
                    
                if wotrans and woadv:
                    loss = bce_loss
                
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} | Trans Loss: {epoch_tran_loss/n_batches:.6f}'
                        f'| Adv Loss: {epoch_adv_loss/n_batches:.6f} | BCE Loss: {epoch_bce_loss / n_batches:.6f}')

        self.train_time = time.time() - start_time
        logger.info('Training Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished training.')

        return self.net

    def test(self, dataset: BaseADDataset, net: BaseNet, wotrans: bool):
        logger = logging.getLogger()

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set device for network
        self.net = net.to(self.device)

        # Testing
        logger.info('Starting testing...')
        epoch_loss = 0.0
        n_batches = 0
        start_time = time.time()
        idx_label_score = []
        self.net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, semi_targets, idx = data
                # known_idx = np.argwhere(np.isin(semi_targets, [-1,1])).flatten()
                inputs = inputs.to(self.device)
                semi_targets = (-2*labels+1).to(self.device)
                labels = labels.to(self.device).to(torch.float)
                idx = idx.to(self.device)

                tran_outputs = self.net(inputs)
                dist = torch.sum((tran_outputs - self.c) ** 2, dim=1)
                tran_losses = dist #torch.where(semi_targets == 0, dist, self.eta * ((dist + self.eps) ** semi_targets.float()))
                tran_loss = torch.mean(tran_losses)
                scores = dist
                
                pre_logits = self.net.pred_adv(tran_outputs)
                pre_logits = torch.squeeze(pre_logits, dim = 1)
                bce_losses = F.binary_cross_entropy_with_logits(pre_logits, labels, reduction='none')
                bce_loss = torch.mean(bce_losses)
                
                loss = tran_loss + bce_loss
                
                pre_scores = torch.sigmoid(pre_logits)
                
                final_scores = scores + pre_scores-0.5
                
                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist(),
                                            pre_scores.cpu().data.numpy().tolist(),
                                            final_scores.cpu().data.numpy().tolist())) 

                epoch_loss += loss.item()
                n_batches += 1

        self.test_time = time.time() - start_time
        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, cluster_scores, prediction_scores, scores = zip(*idx_label_score)
        labels = np.array(labels)
        cluster_scores = np.array(cluster_scores)
        prediction_scores = np.array(prediction_scores)
        scores = np.array(scores)
        self.test_auc_cluster = average_precision_score(labels, cluster_scores) ## default average = 'macro'
        self.test_auc_pred = average_precision_score(labels, prediction_scores)
        self.test_auc = average_precision_score(labels, scores)
        
        matrix_scores = prediction_scores
        
        if not wotrans:
            matrix_scores = scores
        
        p, r, thresholds = precision_recall_curve(labels, matrix_scores)
        maxindex = (r+p).tolist().index(max(r+p))
        threshold = thresholds[maxindex]
        y_pred = np.where(scores >= threshold, 1, 0)
        cal_r = classification_report(y_true = labels, y_pred = y_pred)
        con_m = confusion_matrix(y_true = labels, y_pred = y_pred)
        
        # Log results
        logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        logger.info('Test PR_AUC (based on cluster): {:.2f}%'.format(100. * self.test_auc_cluster))
        logger.info('Test PR_AUC (based on prediction): {:.2f}%'.format(100. * self.test_auc_pred))
        logger.info('Test PR_AUC: {:.2f}%'.format(100. * self.test_auc))
        logger.info('Classification report:\n{}'.format(cal_r))
        logger.info('Confusion matrix:\n{}'.format(con_m))
        logger.info('Test Time: {:.3f}s'.format(self.test_time))
        logger.info('Finished testing.')

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device)
        self.net = net
        self.net.eval()
        with torch.no_grad():
            for data in train_loader:
                # get the inputs of the batch
                inputs, _, semi_targets, _, _ = data # rests of inputs doesn't have labels
                known_idx = np.argwhere(np.isin(semi_targets, 1)).flatten() # known normal
                inputs = inputs.to(self.device)
                outputs = self.net(inputs[known_idx])
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c
    
    def one_class_adv_loss(self, x_train_data):
        """Computes the adversarial loss:
        1) Sample points initially at random around the positive training
            data points
        2) Gradient ascent to find the most optimal point in set N_i(r) 
            classified as +ve (label=0). This is done by maximizing 
            the CE loss wrt label 0
        3) Project the points between spheres of radius R and gamma * R 
            (set N_i(r))
        4) Pass the calculated adversarial points through the model, 
            and calculate the CE loss wrt target class 0
        
        Parameters
        ----------
        x_train_data: Batch of data to compute loss on.
        """
        batch_size = len(x_train_data)
        # Randomly sample points around the training data
        # We will perform SGD on these to find the adversarial points

        x_adv = torch.randn(x_train_data.shape).to(self.device).detach().requires_grad_()
        x_adv_sampled = x_adv + x_train_data
        for step in range(int(self.ascent_num_steps)):
            with torch.enable_grad():

                new_targets = torch.ones(batch_size, 1).to(self.device)
                new_targets = torch.squeeze(new_targets)
                new_targets = new_targets.to(torch.float)
                
                tran_outputs = self.net(x_adv_sampled)
                logits = self.net.pred_adv(tran_outputs)
                logits = torch.squeeze(logits, dim = 1)
                new_loss = F.binary_cross_entropy_with_logits(logits, new_targets)

                grad = torch.autograd.grad(new_loss, [x_adv_sampled])[0]
                grad_norm = torch.norm(grad, p=2, dim = tuple(range(1, grad.dim())))
                grad_norm = grad_norm.view(-1, *[1]*(grad.dim()-1))
                grad_normalized = grad/grad_norm 
                
            with torch.no_grad():
                x_adv_sampled.add_(self.ascent_step_size * grad_normalized)

            if (step + 1) % 1==0:
                # Project the normal points to the set N_i(r)
                h = x_adv_sampled - x_train_data
                norm_h = torch.sqrt(torch.sum(h**2, 
                                                dim=tuple(range(1, h.dim()))))
                alpha = torch.clamp(norm_h, self.radius, 
                                    self.gamma * self.radius).to(self.device)
                # Make use of broadcast to project h
                proj = (alpha/norm_h).view(-1, *[1] * (h.dim()-1))
                h = proj * h
                x_adv_sampled = x_train_data + h  #These adv_points are now on the surface of hyper-sphere (around decision boundary)
                
        tran_outputs = self.net(x_adv_sampled)
        adv_pred = self.net.pred_adv(tran_outputs)
        adv_pred = torch.squeeze(adv_pred, dim=1)
        new_targets = torch.ones(batch_size, 1).to(self.device)
        new_targets = torch.squeeze(new_targets)
        new_targets = new_targets.to(torch.float)
        adv_loss = F.binary_cross_entropy_with_logits(adv_pred, new_targets, reduction='none')
        return adv_loss
    

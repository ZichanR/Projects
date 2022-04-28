import json
import torch
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from base.base_dataset import BaseADDataset
from networks.main import build_network, build_autoencoder
from optim.DeepSAD_trainer import DeepSADTrainer
from optim.ae_trainer import AETrainer


class DeepSAD(object):
    """A class for the Deep SAD method.

    Attributes:
        eta: Deep SAD hyperparameter eta (must be 0 < eta).
        c: Hypersphere center c.
        net_name: A string indicating the name of the neural network to use.
        net: The neural network phi.
        trainer: DeepSADTrainer to train a Deep SAD model.
        optimizer_name: A string indicating the optimizer to use for training the Deep SAD network.
        ae_net: The autoencoder network corresponding to phi for network weights pretraining.
        ae_trainer: AETrainer to train an autoencoder in pretraining.
        ae_optimizer_name: A string indicating the optimizer to use for pretraining the autoencoder.
        results: A dictionary to save the results.
        ae_results: A dictionary to save the autoencoder results.
    """

    def __init__(self, args):
        """Inits DeepSAD with hyperparameter eta."""
        
        self.args = args
        self.eta = args.eta
        self.c = None  # hypersphere center c

        self.net_name = None
        self.net = None  # neural network phi

        self.trainer = None
        self.optimizer_name = None

        self.ae_net = None  # autoencoder network for pretraining
        self.ae_trainer = None
        self.ae_optimizer_name = None

        self.results = {
            'train_time': None,
            'test_auc_cluster': None,
            'test_auc_pred': None,
            'test_auc': None,
            'test_time': None,
            'test_scores': None,
        }

        self.ae_results = {
            'train_time': None,
            'test_auc': None,
            'test_time': None
        }

    def set_network(self, net_name):
        """Builds the neural network phi."""
        self.net_name = net_name
        self.net = build_network(net_name)

    def train(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 50,
              lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
              n_jobs_dataloader: int = 0):
        """Trains the Deep SAD model on the training data."""

        self.optimizer_name = optimizer_name
        self.trainer = DeepSADTrainer(c=self.c, lamda=self.args.lamda, radius=self.args.lamda, gamma=self.args.lamda, 
                                      eta=self.eta, optimizer_name=optimizer_name, lr=lr, n_epochs=n_epochs,
                                      lr_milestones=lr_milestones, ascent_step_size=self.args.ascent_step_size, 
                                      ascent_num_steps=self.args.ascent_step_size, batch_size=batch_size, weight_decay=weight_decay,
                                      device=device, n_jobs_dataloader=n_jobs_dataloader)
        # Get the model
        self.net = self.trainer.train(dataset, self.net, self.args.wotrans,self.args.woadv)
        self.results['train_time'] = self.trainer.train_time
        self.c = self.trainer.c.cpu().data.numpy().tolist()  # get as list

    def test(self, dataset: BaseADDataset, device: str = 'cuda', n_jobs_dataloader: int = 0):
        """Tests the Deep SAD model on the test data."""

        if self.trainer is None:
            self.trainer = DeepSADTrainer(self.c, self.eta, device=device, n_jobs_dataloader=n_jobs_dataloader)

        self.trainer.test(dataset, self.net, self.args.wotrans)

        # Get results
        self.results['test_auc_cluster'] = self.trainer.test_auc_cluster
        self.results['test_auc_pred'] = self.trainer.test_auc_pred
        self.results['test_auc'] = self.trainer.test_auc
        # self.results['test_fpr'] = self.trainer.test_fpr.tolist()
        # self.results['test_tpr'] = self.trainer.test_tpr.tolist()
        # self.results['test_thresholds'] = self.trainer.test_thresholds.tolist()
        # self.results['test_prec'] = self.trainer.test_prec
        # self.results['test_recall'] = self.trainer.test_recall
        # self.results['test_fscore'] = self.trainer.test_fscore
        # self.results['test_cal_r'] = self.trainer.test_cal_r
        # self.results['test_con_m'] = self.trainer.test_con_m.tolist()        
        self.results['test_time'] = self.trainer.test_time
        self.results['test_scores'] = self.trainer.test_scores
        
    def visualization(self, dataset: BaseADDataset, img_file: str ):
        """Trains the Deep SAD model on the training data."""

        train_data = dataset.train_set
        train_feature = self.net(train_data.data)
        # if self.c is None:
        #     train_loader, _ = dataset.loaders(batch_size=self.args.batch_size, num_workers=self.args.n_jobs_dataloader)
        #     self.c = self.trainer.init_center_c(train_loader, self.net)
        # dist = torch.sum((train_feature - torch.tensor(self.c)) ** 2, dim=1)
        # dist = dist.cpu().detach().numpy() 
        train_tsne = TSNE(n_components=2)
        train_Y = train_tsne.fit_transform(train_feature.cpu().detach().numpy())
        color_code = ['skyblue', 'mediumvioletred']
        labels = train_data.real_targets.cpu().detach().numpy().tolist()
        color = [color_code[i] for i in labels]
        plt.figure(figsize=(40, 30))
        plt.scatter(train_Y[:, 0], train_Y[:, 1], c=color) #, s=dist)
        plt.show()
        plt.savefig(img_file)
        plt.clf()
        
        
    def pretrain(self, dataset: BaseADDataset, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 100,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        """Pretrains the weights for the Deep SAD network phi via autoencoder."""

        # Set autoencoder network
        self.ae_net = build_autoencoder(self.net_name)

        # Train
        self.ae_optimizer_name = optimizer_name
        self.ae_trainer = AETrainer(optimizer_name, lr=lr, n_epochs=n_epochs, lr_milestones=lr_milestones,
                                    batch_size=batch_size, weight_decay=weight_decay, device=device,
                                    n_jobs_dataloader=n_jobs_dataloader)
        self.ae_net = self.ae_trainer.train(dataset, self.ae_net, futurepre=self.args.futurepre)

        # Get train results
        self.ae_results['train_time'] = self.ae_trainer.train_time

        # Test
        self.ae_trainer.test(dataset, self.ae_net, futurepre=self.args.futurepre)

        # Get test results
        self.ae_results['test_auc'] = self.ae_trainer.test_auc
        self.ae_results['test_time'] = self.ae_trainer.test_time

        # Initialize Deep SAD network weights from pre-trained encoder
        self.init_network_weights_from_pretraining()

    def init_network_weights_from_pretraining(self):
        """Initialize the Deep SAD network weights from the encoder weights of the pretraining autoencoder."""

        net_dict = self.net.state_dict()
        ae_net_dict = self.ae_net.state_dict()

        # Filter out decoder network keys
        ae_net_dict = {k: v for k, v in ae_net_dict.items() if k in net_dict}
        # Overwrite values in the existing state_dict
        net_dict.update(ae_net_dict)
        # Load the new state_dict
        self.net.load_state_dict(net_dict)

    def save_model(self, export_model, save_ae=True):
        """Save Deep SAD model to export_model."""

        net_dict = self.net.state_dict()
        pred_dict = self.net.pred_adv.state_dict()
        ae_net_dict = self.ae_net.state_dict() if save_ae else None

        torch.save({'c': self.c,
                    'pred_dict': pred_dict,
                    'net_dict': net_dict,
                    'ae_net_dict': ae_net_dict}, export_model)

    def load_model(self, model_path, load_ae=False, map_location='cpu'):
        """Load Deep SAD model from model_path."""

        model_dict = torch.load(model_path, map_location=map_location)

        self.c = model_dict['c']
        self.net.load_state_dict(model_dict['net_dict'])
        self.net.pred_adv.load_state_dict(model_dict['pred_dict'])

        # load autoencoder parameters if specified
        if load_ae:
            if self.ae_net is None:
                self.ae_net = build_autoencoder(self.net_name)
            self.ae_net.load_state_dict(model_dict['ae_net_dict'])
            
    def load_model_from_pretrain(self, model_path, map_location='cpu'):
        """Load Deep SAD model from model_path."""

        model_dict = torch.load(model_path, map_location=map_location)

        if self.ae_net is None:
            self.ae_net = build_autoencoder(self.net_name)
        self.ae_net.load_state_dict(model_dict['ae_net_dict'])

        # Initialize Deep SAD network weights from pre-trained encoder
        self.init_network_weights_from_pretraining()

    def save_results(self, export_json):
        """Save results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.results, fp)

    def save_ae_results(self, export_json):
        """Save autoencoder results dict to a JSON-file."""
        with open(export_json, 'w') as fp:
            json.dump(self.ae_results, fp)

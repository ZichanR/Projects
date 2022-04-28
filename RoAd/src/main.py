import click
import argparse
import os
import torch
import logging
import random
import numpy as np

from utils.config import Config
from utils.visualization.plot_images_grid import plot_images_grid
from utils.utils import prepare_folders
from DeepSAD import DeepSAD
from datasets.main import load_dataset

import warnings
warnings.filterwarnings("ignore")

################################################################################
# Settings
################################################################################
# basic infor
parser = argparse.ArgumentParser(description='PyTorch RoAd, a fully deep method for semi-supervised learning-based anomaly detection against noisy labels.')
parser.add_argument('--dataset_name', default='VisAnaML', 
                    choices=['VisAnaML', 'arrhythmia', 'cardio', 'satellite', 'satimage-2', 'shuttle', 'thyroid'],
                    help='dataset setting')
parser.add_argument('--net_name', metavar='ARCH', default='VisAnaML_LSTM',
                    choices=['VisAnaML_LSTM', 'arrhythmia_mlp', 'cardio_mlp', 'satellite_mlp', 
                             'satimage-2_mlp', 'shuttle_mlp','thyroid_mlp'])
parser.add_argument('--wotrans',type=int, default=1)
parser.add_argument('--woadv',type=int, default=0)
parser.add_argument('--visualization',type=int, default=0)
parser.add_argument('--hd', type=int, default=64, metavar='N',
                    help='Num hidden nodes for LSTM model')
parser.add_argument('--xp_path',type=str, default='../log')
parser.add_argument('--data_path',type=str, default='../data')
parser.add_argument('--fig_path', type=str, default='../imgs')
parser.add_argument('--load_config', type=str, default=None, 
                    help='Config JSON-file path (default: None).')
parser.add_argument('--load_model', default='', type=str, metavar='PATH', 
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--load_model_from_pretrain', default='', type=str, metavar='PATH', 
                    help='path to pretrianed model (default: none)')
parser.add_argument('--device', type=str, default='cuda', 
                    help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
# parser.add_argument('--anomaly_ratio', type=float, default=0, choices=[0, 0.05, 0.1, 0.2], 
#                     help="Anomaly ratio set in datasets, when set to 0 meaning only normal data are used during training.")
parser.add_argument('--num_threads', type=int, default=0,
                    help='Number of threads used for parallelizing CPU operations. 0 means that all resources are used.')
parser.add_argument('--n_jobs_dataloader', type=int, default=0,
                    help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
parser.add_argument('--exp_str', default='0', type=str, 
                    help='number to indicate which experiment it is')

# pretraining settings
parser.add_argument('--pretrain', type=int, default=1, choices=[0,1], 
                    help='Pretrain neural network parameters via autoencoder.')
parser.add_argument('--ae_optimizer_name', choices=['adam'], default='adam', 
                    help='Name of the optimizer to use for autoencoder pretraining.')
parser.add_argument('--ae_lr', type=float, default=0.001,
                    help='Initial learning rate for autoencoder pretraining. Default=0.001')
parser.add_argument('--ae_n_epochs', type=int, default=150, 
                    help='Number of epochs to train autoencoder.')
parser.add_argument('--ae_lr_milestone', type=int, nargs='+', default=[50], 
                    help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
parser.add_argument('--ae_batch_size', type=int, default=128, 
                    help='Batch size for mini-batch autoencoder training.')
parser.add_argument('--ae_weight_decay', type=float, default=0.5e-3, 
                    help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
parser.add_argument('--traintestsplit', type=str, default='date', choices=['date','random'])
parser.add_argument('--futurepre', type=int, default=0)

# about noisy label settings
parser.add_argument('--mislabel_type', type=str, default='agnostic')
parser.add_argument('--mislabel_ratio', type=float, default=0.5)
parser.add_argument('--rand-number', type=int, default=0,
                    help="Ratio for number of facilities.")

# about semi-supervised learning settings
parser.add_argument('--eta', type=float, default=1.0, 
                    help='hyperparameter eta (must be 0 < eta).')
parser.add_argument('--ratio_unlabel', type=float, default=0.0, choices=[0.0,0.2,0.5,0.8],
                    help='Ratio of unknown label training examples.')
parser.add_argument('--n_pollution', type=int, default=0, choices=[0,100,200,400,800],
                    help='Pollution of unlabeled training data with unknown (unlabeled) anomalies.')
# parser.add_argument('--normal_class', type=int, default=0,
#               help='Specify the normal class of the dataset (all other classes are considered anomalous).')
# parser.add_argument('--known_outlier_class', type=int, default=1,
#               help='Specify the known outlier class of the dataset for semi-supervised anomaly detection.')
# parser.add_argument('--n_known_outlier_classes', type=int, default=0,
#               help='Number of known outlier classes.'
#                    'If 0, no anomalies are known.'
#                    'If 1, outlier class as specified in --known_outlier_class option.'
#                    'If > 1, the specified number of outlier classes will be sampled at random.')
parser.add_argument('--seed', type=int, default=1, 
                    help='Set seed. If -1, use randomization.')
parser.add_argument('--optimizer_name',type=str, default='adam', choices=['adam'],
                    help='Name of the optimizer to use for Deep SVDD network training.')
parser.add_argument('--n_epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', default=0.0001, type=float,
                    metavar='LR', 
                    help='Initial learning rate for Deep SAD network training. Default=0.001', dest='lr')
parser.add_argument('--weight_decay', default=0.5e-6, type=float,metavar='W', 
                    help='Weight decay (L2 penalty) hyperparameter for Deep SVDD objective.',dest='weight_decay')
parser.add_argument('--batch_size', default=128, type=int,metavar='N',
                    help='Batch size for mini-batch training.')
parser.add_argument('--lr_milestone', type=int, nargs='+', default=[50],
                    help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')

# adv training setting
parser.add_argument('--ascent_step_size', type=float, default=0.001, metavar='LR',
                    help='step size of gradient ascent') 
parser.add_argument('--ascent_num_steps', type=int, default=100, metavar='N',
                    help='Number of gradient ascent steps')  
parser.add_argument('--radius', type=float, default=8, metavar='N',
                    help='radius corresponding to the definition of set N_i(r)')
parser.add_argument('--lamda', type=float, default=1, metavar='N',
                    help='Weight to the adversarial loss')
parser.add_argument('--gamma', type=float, default=1.0, metavar='N',
                    help='r to gamma * r projection for the set N_i(r)')

def main():
    """
    Deep SAD, a method for deep semi-supervised anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    # Get configuration
    args = parser.parse_args()
    args.wotrans, args.woadv, args.pretrain, args.futurepre = bool(args.wotrans), bool(args.woadv), bool(args.pretrain), bool(args.futurepre)
    if not args.ratio_unlabel:
        args.n_pollution = 0
    
    # prepare result folders and paths
    if args.visualization:
        args.store_name = '_'.join([args.dataset_name, 'TSNE', args.exp_str])
    else:
        args.store_name = '_'.join([args.dataset_name, str(args.ratio_unlabel), str(args.n_pollution), 
                                    args.mislabel_type, str(args.mislabel_ratio), args.exp_str])
    prepare_folders(args)
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = os.path.join(args.xp_path, args.store_name, 'log.txt')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    
    # Print paths
    logger.info('Log file is %s' % log_file)
    logger.info('Data path is %s' % args.data_path)
    logger.info('Export path is %s' % args.xp_path)

    # Print experimental setup
    logger.info('Dataset: %s' % args.dataset_name)
    logger.info('Ratio of unlabeled samples: %.2f' % args.ratio_unlabel)
    logger.info('Pollution of anomalies in unlabeled train data: %d' % args.n_pollution)
    logger.info('Ratio of mislabeled train data: %.2f' % args.mislabel_ratio)
    logger.info('Mislabel type in train data: %s' % args.mislabel_type)    
    logger.info('Network: %s' % args.net_name)

    # If specified, load experiment config from JSON-file
    # if args.load_config:
    #     cfg = Config({'load_config':args.load_config})
    #     args = cfg.load_config(import_json=args.load_config)
    #     logger.info('Loaded configuration from %s.' % args.load_config)

    # Print model configuration
    logger.info('Eta-parameter: %.2f' % args.eta)
    logger.info('Lambda-parameter: %.2f' % args.lamda)
    logger.info('Gamma-parameter: %.2f' % args.gamma)
    logger.info('Radius: %.2f' % args.radius)
    logger.info('Number of gradient ascent steps during adversary generation: %d' % args.ascent_num_steps)
    logger.info('Step size of gradient ascent during adversary generation: %.4f' % args.ascent_step_size)

    # Set seed
    if args.seed != -1:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        logger.info('Set seed to %d.' % args.seed)

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        args.device = 'cpu'
    # Set the number of threads used for parallelizing CPU operations
    if args.num_threads > 0:
        torch.set_num_threads(args.num_threads)
    logger.info('Computation device: %s' % args.device)
    logger.info('Number of threads: %d' % args.num_threads)
    logger.info('Number of dataloader workers: %d' % args.n_jobs_dataloader)

    # Load data
    dataset = load_dataset(args)
    logger.info('Train test split (8/2) according to: %s' % args.traintestsplit)
    
    # Initialize DeepSAD model and set neural network phi
    deepSAD = DeepSAD(args)
    deepSAD.set_network(args.net_name)

    # If specified, load Deep SAD model (center c, network weights, and possibly autoencoder weights)
    if args.load_model:
        deepSAD.load_model(model_path=args.load_model, load_ae=True, map_location=args.device)
        logger.info('Loading model from %s.' % args.load_model)
        if args.visualization:
            logger.info('Visualizing latent features from loaded model.')
            file = os.path.join(args.xp_path, args.store_name, '{}_tsne.png'.format(args.exp_str))
            deepSAD.visualization(dataset,file)
            return
            
    if args.load_model_from_pretrain:
        deepSAD.load_model_from_pretrain(model_path=args.load_model_from_pretrain, map_location=args.device)
        logger.info('Loading model from pretrained model %s.' % args.load_model_from_pretrain)
    
    logger.info('Pretraining: %s' % args.pretrain)
    if args.pretrain:
        # Log pretraining details
        logger.info('Pretraining with future prediction: %s' % args.futurepre)
        logger.info('Pretraining optimizer: %s' % args.ae_optimizer_name)
        logger.info('Pretraining learning rate: %g' % args.ae_lr)
        logger.info('Pretraining epochs: %d' % args.ae_n_epochs)
        logger.info('Pretraining learning rate scheduler milestones: %s' % (args.ae_lr_milestone,))
        logger.info('Pretraining batch size: %d' % args.ae_batch_size)
        logger.info('Pretraining weight decay: %g' % args.ae_weight_decay)

        # Pretrain model on dataset (via autoencoder)
        deepSAD.pretrain(dataset,
                         optimizer_name=args.ae_optimizer_name,
                         lr=args.ae_lr,
                         n_epochs=args.ae_n_epochs,
                         lr_milestones=args.ae_lr_milestone,
                         batch_size=args.ae_batch_size,
                         weight_decay=args.ae_weight_decay,
                         device=args.device,
                         n_jobs_dataloader=args.n_jobs_dataloader)

        # Save pretraining results
        deepSAD.save_ae_results(export_json=os.path.join(args.xp_path, args.store_name, 'ae_results.json'))

    # Log training details
    logger.info('Training optimizer: %s' % args.optimizer_name)
    logger.info('Training learning rate: %g' % args.lr)
    logger.info('Training epochs: %d' % args.n_epochs)
    logger.info('Training learning rate scheduler milestones: %s' % (args.lr_milestone,))
    logger.info('Training batch size: %d' % args.batch_size)
    logger.info('Training weight decay: %g' % args.weight_decay)
    logger.info('Training without trans loss: %s' % args.wotrans)
    logger.info('Training without adv loss: %s' % args.woadv)
    # Train model on dataset
    deepSAD.train(dataset,
                    optimizer_name=args.optimizer_name,
                    lr=args.lr,
                    n_epochs=args.n_epochs,
                    lr_milestones=args.lr_milestone,
                    batch_size=args.batch_size,
                    weight_decay=args.weight_decay,
                    device=args.device,
                    n_jobs_dataloader=args.n_jobs_dataloader)

    # Test model
    deepSAD.test(dataset, device=args.device, n_jobs_dataloader=args.n_jobs_dataloader)

    # Save results, model, and configuration
    results_path = os.path.join(args.xp_path, args.store_name, 'results.json')
    checkpoint_path = os.path.join(args.xp_path, args.store_name, 'model.tar')
    config_path = os.path.join(args.xp_path, args.store_name, 'config.json')
    deepSAD.save_results(export_json=results_path)
    deepSAD.save_model(export_model=checkpoint_path)
    logger.info('Model saved as: %s' % checkpoint_path)
    Config(vars(args)).save_config(export_json=config_path)    

    
    # # Plot most anomalous and most normal test samples
    # indices, labels, scores = zip(*deepSAD.results['test_scores'])
    # indices, labels, scores = np.array(indices), np.array(labels), np.array(scores)
    # idx_all_sorted = indices[np.argsort(scores)]  # from lowest to highest score
    # idx_normal_sorted = indices[labels == 0][np.argsort(scores[labels == 0])]  # from lowest to highest score

    # if args.dataset_name in ('mnist', 'fmnist', 'cifar10'):

    #     if args.dataset_name in ('mnist', 'fmnist'):
    #         X_all_low = dataset.test_set.data[idx_all_sorted[:32], ...].unsqueeze(1)
    #         X_all_high = dataset.test_set.data[idx_all_sorted[-32:], ...].unsqueeze(1)
    #         X_normal_low = dataset.test_set.data[idx_normal_sorted[:32], ...].unsqueeze(1)
    #         X_normal_high = dataset.test_set.data[idx_normal_sorted[-32:], ...].unsqueeze(1)

    #     if args.dataset_name == 'cifar10':
    #         X_all_low = torch.tensor(np.transpose(dataset.test_set.data[idx_all_sorted[:32], ...], (0,3,1,2)))
    #         X_all_high = torch.tensor(np.transpose(dataset.test_set.data[idx_all_sorted[-32:], ...], (0,3,1,2)))
    #         X_normal_low = torch.tensor(np.transpose(dataset.test_set.data[idx_normal_sorted[:32], ...], (0,3,1,2)))
    #         X_normal_high = torch.tensor(np.transpose(dataset.test_set.data[idx_normal_sorted[-32:], ...], (0,3,1,2)))

    #     plot_images_grid(X_all_low, export_img=os.path.join(args.xp_path, args.store_name, 'all_low'), padding=2)
    #     plot_images_grid(X_all_high, export_img=os.path.join(args.xp_path, args.store_name, 'all_high'), padding=2)
    #     plot_images_grid(X_normal_low, export_img=os.path.join(args.xp_path, args.store_name, 'normals_low'), padding=2)
    #     plot_images_grid(X_normal_high, export_img=os.path.join(args.xp_path, args.store_name, 'normals_high'), padding=2)


if __name__ == '__main__':
    main()

################################################################################
# Settings
################################################################################
# @click.command()
# @click.argument('dataset_name', type=click.Choice(['mnist', 'fmnist', 'cifar10', 'arrhythmia', 'cardio', 'satellite',
#                                                    'satimage-2', 'shuttle', 'thyroid']))
# @click.argument('net_name', type=click.Choice(['mnist_LeNet', 'fmnist_LeNet', 'cifar10_LeNet', 'arrhythmia_mlp',
#                                                'cardio_mlp', 'satellite_mlp', 'satimage-2_mlp', 'shuttle_mlp',
#                                                'thyroid_mlp']))
# @click.argument('xp_path', type=click.Path(exists=True))
# @click.argument('data_path', type=click.Path(exists=True))
# @click.option('--load_config', type=click.Path(exists=True), default=None,
#               help='Config JSON-file path (default: None).')
# @click.option('--load_model', type=click.Path(exists=True), default=None,
#               help='Model file path (default: None).')
# @click.option('--eta', type=float, default=1.0, help='Deep SAD hyperparameter eta (must be 0 < eta).')
# @click.option('--ratio_known_normal', type=float, default=0.0,
#               help='Ratio of known (labeled) normal training examples.')
# @click.option('--ratio_known_outlier', type=float, default=0.0,
#               help='Ratio of known (labeled) anomalous training examples.')
# @click.option('--ratio_pollution', type=float, default=0.0,
#               help='Pollution ratio of unlabeled training data with unknown (unlabeled) anomalies.')
# @click.option('--device', type=str, default='cuda', help='Computation device to use ("cpu", "cuda", "cuda:2", etc.).')
# @click.option('--seed', type=int, default=-1, help='Set seed. If -1, use randomization.')
# @click.option('--optimizer_name', type=click.Choice(['adam']), default='adam',
#               help='Name of the optimizer to use for Deep SAD network training.')
# @click.option('--lr', type=float, default=0.001,
#               help='Initial learning rate for Deep SAD network training. Default=0.001')
# @click.option('--n_epochs', type=int, default=50, help='Number of epochs to train.')
# @click.option('--lr_milestone', type=int, default=0, multiple=True,
#               help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
# @click.option('--batch_size', type=int, default=128, help='Batch size for mini-batch training.')
# @click.option('--weight_decay', type=float, default=1e-6,
#               help='Weight decay (L2 penalty) hyperparameter for Deep SAD objective.')
# @click.option('--pretrain', type=bool, default=True,
#               help='Pretrain neural network parameters via autoencoder.')
# @click.option('--ae_optimizer_name', type=click.Choice(['adam']), default='adam',
#               help='Name of the optimizer to use for autoencoder pretraining.')
# @click.option('--ae_lr', type=float, default=0.001,
#               help='Initial learning rate for autoencoder pretraining. Default=0.001')
# @click.option('--ae_n_epochs', type=int, default=100, help='Number of epochs to train autoencoder.')
# @click.option('--ae_lr_milestone', type=int, default=0, multiple=True,
#               help='Lr scheduler milestones at which lr is multiplied by 0.1. Can be multiple and must be increasing.')
# @click.option('--ae_batch_size', type=int, default=128, help='Batch size for mini-batch autoencoder training.')
# @click.option('--ae_weight_decay', type=float, default=1e-6,
#               help='Weight decay (L2 penalty) hyperparameter for autoencoder objective.')
# @click.option('--num_threads', type=int, default=0,
#               help='Number of threads used for parallelizing CPU operations. 0 means that all resources are used.')
# @click.option('--n_jobs_dataloader', type=int, default=0,
#               help='Number of workers for data loading. 0 means that the data will be loaded in the main process.')
# @click.option('--normal_class', type=int, default=0,
#               help='Specify the normal class of the dataset (all other classes are considered anomalous).')
# @click.option('--known_outlier_class', type=int, default=1,
#               help='Specify the known outlier class of the dataset for semi-supervised anomaly detection.')
# @click.option('--n_known_outlier_classes', type=int, default=0,
#               help='Number of known outlier classes.'
#                    'If 0, no anomalies are known.'
#                    'If 1, outlier class as specified in --known_outlier_class option.'
#                    'If > 1, the specified number of outlier classes will be sampled at random.')
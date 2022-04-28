from .visanaml import VisAnaML_Dataset
from .fmnist import FashionMNIST_Dataset
from .cifar10 import CIFAR10_Dataset
from .odds import ODDSADDataset


def load_dataset(args):
    """Loads the dataset."""

    implemented_datasets = ('VisAnaML', 'fmnist', 'cifar10',
                            'arrhythmia', 'cardio', 'satellite', 'satimage-2', 'shuttle', 'thyroid')
    assert args.dataset_name in implemented_datasets

    dataset = None

    if args.dataset_name == 'VisAnaML':
        dataset = VisAnaML_Dataset(root=args.data_path,
                                args=args,
                                ratio_unlabel=args.ratio_unlabel, 
                                n_pollution=args.n_pollution)

    if args.dataset_name == 'fmnist':
        dataset = FashionMNIST_Dataset(root=args.data_path,
                                       normal_class=args.normal_class,
                                       known_outlier_class=args.known_outlier_class,
                                       n_known_outlier_classes=args.n_known_outlier_classes,
                                       ratio_known_normal=args.ratio_known_normal,
                                       ratio_known_outlier=args.ratio_known_outlier,
                                       ratio_pollution=args.ratio_pollution)

    if args.dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=args.data_path,
                                  normal_class=args.normal_class,
                                  known_outlier_class=args.known_outlier_class,
                                  n_known_outlier_classes=args.n_known_outlier_classes,
                                  ratio_known_normal=args.ratio_known_normal,
                                  ratio_known_outlier=args.ratio_known_outlier,
                                  ratio_pollution=args.ratio_pollution)

    if args.dataset_name in ('arrhythmia', 'cardio', 'satellite', 'satimage-2', 'shuttle', 'thyroid'):
        dataset = ODDSADDataset(root=args.data_path,
                                dataset_name=args.dataset_name,
                                n_known_outlier_classes=args.n_known_outlier_classes,
                                ratio_known_normal=args.ratio_known_normal,
                                ratio_known_outlier=args.ratio_known_outlier,
                                ratio_pollution=args.ratio_pollution,
                                random_state=args.random_state)

    return dataset

# -*- coding: utf-8 -*-
"""
Utility functions for IEEG data processing, model checkpoint handling, and data loaders.
"""
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'src')))
from ieeg_dataset import IeegDataset
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from typing import Tuple, Union
from torch.utils.data import random_split
from collections import Counter 


def get_class_distribution(dataset):
    """
    Calculate the distribution of classes in the dataset.

    Args:
        dataset (Dataset): The dataset for which to calculate the class distribution.

    Returns:
        dict: A dictionary with class labels as keys and their corresponding counts as values.
    """
    labels = [dataset[i][1].item() for i in range(len(dataset))]
    class_counts = Counter(labels)
    total_count = len(labels)
    class_distribution = {label: count / total_count for label, count in class_counts.items()}
    return class_distribution

def get_loaders_no_sss(train_dir: str, test_dir: str, batch_size: int, num_workers: int, 
                       pin_memory: bool = True, with_val_loader: bool = False, 
                       val_size: float = 0.5, seq_length: int = 5000, 
                       model_type: str = "mlp", label_encoder=None) -> Union[
                           Tuple[DataLoader, DataLoader, dict, dict], 
                           Tuple[DataLoader, DataLoader, DataLoader, dict, dict, dict]]:
    """
    Creates data loaders for training, testing, and optionally validation from IEEG dataset without using Stratified Shuffle Split (SSS).

    Args:
        train_dir (str): Directory containing the training data files.
        test_dir (str): Directory containing the test data files.
        batch_size (int): Number of samples per batch to load.
        num_workers (int): Number of subprocesses to use for data loading.
        pin_memory (bool, optional): If True, the data loader will copy tensors into CUDA pinned memory before returning them. Default is True.
        with_val_loader (bool, optional): If True, a validation data loader is also returned. Default is False.
        val_size (float, optional): Proportion of the test set to include in the validation split. Default is 0.1.
        seq_length (int, optional): Length of each signal sequence. Default is 5000.
        model_type (str, optional): Specifies the type of model (e.g., "mlp", "cnn", "seq"). Default is "mlp".
        label_encoder (LabelEncoder, optional): Label encoder to use. Default is None.

    Returns:
        Union[Tuple[DataLoader, DataLoader, dict, dict], 
              Tuple[DataLoader, DataLoader, DataLoader, dict, dict, dict]]:
        Returns train and test data loaders and their class distributions. If `with_val_loader` is True, also returns a validation data loader and its class distribution.
    """
    train_dataset = IeegDataset(train_dir, seq_length=seq_length, model_type=model_type, label_encoder=label_encoder)
    test_dataset = IeegDataset(test_dir, seq_length=seq_length, model_type=model_type, label_encoder=label_encoder)
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    if with_val_loader:
        val_len = int(len(test_dataset) * val_size)
        test_len = len(test_dataset) - val_len
        test_dataset, val_dataset = random_split(test_dataset, [test_len, val_len])
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True, drop_last=True)
        val_distribution = get_class_distribution(val_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True, drop_last=True)

    train_distribution = get_class_distribution(train_dataset)
    test_distribution = get_class_distribution(test_dataset)

    if with_val_loader:
        print(f"Train dataset length: {len(train_dataset)}, Val dataset length: {len(val_dataset)}, Test dataset length: {len(test_dataset)}")
        return train_loader, val_loader, test_loader, train_distribution, val_distribution, test_distribution

    print(f"Train dataset length: {len(train_dataset)}, Test dataset length: {len(test_dataset)}")
    return train_loader, test_loader, train_distribution, test_distribution

def get_loaders_noss(data_dir: str, batch_size: int, num_workers: int, 
                pin_memory: bool = True, with_val_loader: bool = False, 
                test_size: float = 0.2, val_size: float = 0.1, seq_length: int = 5000, 
                model_type: str = "mlp") -> Union[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader, DataLoader]]:
    """
    Creates data loaders for training, testing, and optionally validation from IEEG dataset.

    Args:
        data_dir (str): Directory containing the data files.
        batch_size (int): Number of samples per batch to load.
        num_workers (int): Number of subprocesses to use for data loading.
        pin_memory (bool, optional): If True, the data loader will copy tensors into CUDA pinned memory before returning them. Default is True.
        with_val_loader (bool, optional): If True, a validation data loader is also returned. Default is False.
        test_size (float, optional): Proportion of the dataset to include in the test split. Default is 0.2.
        val_size (float, optional): Proportion of the training set to include in the validation split. Default is 0.1.
        seq_length (int, optional): Length of each signal sequence. Default is 5000.
        model_type (str, optional): Specifies the type of model (e.g., "mlp", "cnn", "seq"). Default is "mlp".

    Returns:
        Union[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader, DataLoader]]:
        Returns train and test data loaders. If `with_val_loader` is True, also returns a validation data loader.
    """
    
    dataset = IeegDataset(data_dir, seq_length=seq_length, model_type=model_type)
    print(f"Total dataset size: {len(dataset)}")

    # Calculate the number of samples for each split
    test_size_int = int(test_size * len(dataset))
    train_size_int = len(dataset) - test_size_int

    # Random split
    train_dataset, test_dataset = random_split(dataset, [train_size_int, test_size_int])
    print(f"Train dataset size: {len(train_dataset)}, Test dataset size: {len(test_dataset)}")

    if with_val_loader:
        # Calculate the number of samples for the validation split
        val_size_int = int(val_size * len(train_dataset))
        train_size_int = len(train_dataset) - val_size_int

        # Random split of the training set into training and validation sets
        train_dataset, val_dataset = random_split(train_dataset, [train_size_int, val_size_int])
        print(f"Train dataset size after val split: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")

        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True, drop_last=True)

    if model_type == "seq":
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True, drop_last=True)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True, drop_last=True)

    if with_val_loader:
        print(f"Train dataset length: {len(train_dataset)}, Val dataset length: {len(val_dataset)}, Test dataset length: {len(test_dataset)}")
        return train_loader, val_loader, test_loader, dataset

    print(f"Train dataset length: {len(train_dataset)}, Test dataset length: {len(test_dataset)}")
    return train_loader, test_loader, dataset

def get_loaders(data_dir: str, batch_size: int, num_workers: int, 
                pin_memory: bool = True, with_val_loader: bool = False, 
                test_size: float = 0.2, val_size: float = 0.1, seq_length: int = 5000, 
                model_type: str = "mlp") -> Union[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader, DataLoader]]:
    """
    Creates data loaders for training, testing, and optionally validation from IEEG dataset.

    Args:
        data_dir (str): Directory containing the data files.
        batch_size (int): Number of samples per batch to load.
        num_workers (int): Number of subprocesses to use for data loading.
        pin_memory (bool, optional): If True, the data loader will copy tensors into CUDA pinned memory before returning them. Default is True.
        with_val_loader (bool, optional): If True, a validation data loader is also returned. Default is False.
        test_size (float, optional): Proportion of the dataset to include in the test split. Default is 0.2.
        val_size (float, optional): Proportion of the training set to include in the validation split. Default is 0.1.
        seq_length (int, optional): Length of each signal sequence. Default is 5000.
        model_type (str, optional): Specifies the type of model (e.g., "mlp", "cnn", "seq"). Default is "mlp".

    Returns:
        Union[Tuple[DataLoader, DataLoader], Tuple[DataLoader, DataLoader, DataLoader]]:
        Returns train and test data loaders. If `with_val_loader` is True, also returns a validation data loader.
    """
    
    dataset = IeegDataset(data_dir, seq_length=seq_length, model_type=model_type)
    print(f"Total dataset size: {len(dataset)}")

    labels = np.array([dataset[i][1].item() for i in range(len(dataset))])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)

    train_index, test_index = next(sss.split(np.zeros(len(labels)), labels))
    print(f"Train indices length: {len(train_index)}, Test indices length: {len(test_index)}")

    train_dataset = Subset(dataset, train_index)
    test_dataset = Subset(dataset, test_index)
    
    if with_val_loader:
        train_labels = np.array([train_dataset[i][1].item() for i in range(len(train_dataset))])
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
        train_idx, val_idx = next(sss_val.split(np.zeros(len(train_labels)), train_labels))
        print(f"Train indices after val split: {len(train_idx)}, Val indices: {len(val_idx)}")

        # Adjust indices to match the train_dataset length
        train_dataset = Subset(dataset, [train_index[i] for i in train_idx])
        val_dataset = Subset(dataset, [train_index[i] for i in val_idx])

        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True, drop_last=True)
    
    if model_type == "seq":
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True, drop_last=True)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True, drop_last=True)

    if with_val_loader:
        print(f"Train dataset length: {len(train_dataset)}, Val dataset length: {len(val_dataset)}, Test dataset length: {len(test_dataset)}")
        return train_loader, val_loader, test_loader, dataset

    print(f"Train dataset length: {len(train_dataset)}, Test dataset length: {len(test_dataset)}")
    return train_loader, test_loader, dataset

def import_checkpoint(checkpoint_path: str, model: torch.nn.Module) -> None:
    """
    Imports model checkpoint from a file.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): Model instance to load the checkpoint into.

    Returns:
        None
    """
    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        print('Checkpoint imported correctly')
    except Exception as e:
        print(f'Error importing checkpoint: {e}')

def save_checkpoint(state: dict, filename: str = 'my_checkpoint.pth.tar') -> None:
    """
    Saves model checkpoint to a file.

    Args:
        state (dict): State dictionary containing model state and other information.
        filename (str, optional): Filename for saving the checkpoint. Default is 'my_checkpoint.pth.tar'.

    Returns:
        None
    """
    try:
        torch.save(state, filename)
        print('Checkpoint saved successfully.')
    except Exception as e:
        print(f'Error saving checkpoint: {e}')

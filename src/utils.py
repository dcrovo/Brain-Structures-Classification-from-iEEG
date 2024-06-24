# -*- coding: utf-8 -*-
"""
Utility functions for IEEG data processing, model checkpoint handling, and data loaders.
"""


from ieeg_dataset import IeegDataset
from torch.utils.data import DataLoader, Subset
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from typing import Tuple, Union

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

        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False, drop_last=True)
    
    if model_type == "seq":
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True, drop_last=True)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False, drop_last=True)

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

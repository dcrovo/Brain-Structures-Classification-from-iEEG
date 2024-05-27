# -*- coding: utf-8 -*-

"""
IEEG Dataset Module
"""

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
import math
import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict

class IeegDataset(Dataset):
    """
    A custom Dataset class for IEEG signals.

    Args:
        data_dir (str): Directory containing the data files.
        seq_length (int, optional): Length of each signal sequence. Default is 5000.
        for_cnn (bool, optional): If True, the data is prepared for CNN input. Default is False.
    """
    def __init__(self, data_dir: str, seq_length: int = 5000, model_type: str = "ann"):
        self.data_dir = data_dir
        self.seq_length = seq_length
        self.model_type= model_type
        self.data = []
        self.labels = []

        # Extract class names from filenames
        self.classes = [f.split('_')[0] for f in os.listdir(self.data_dir)]
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.classes)

        # Load data from CSV files
        self._load_data()
        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels).squeeze(), dtype=torch.long)

    def _load_data(self) -> None:
        """
        Loads the data from the CSV files in the data directory.
        """
        for file in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, file)
            df = pd.read_csv(file_path)

            for column in df.columns: 
                for idx in range(0, math.floor(df.shape[0] / self.seq_length)):
                    signal_window = df[column].values[idx * self.seq_length : (idx + 1) * self.seq_length]
                    class_label = self.label_encoder.transform([file.split('_')[0]])

                    self.data.append(signal_window)
                    self.labels.append(class_label)

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves the sample and label at the given index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The sample and its corresponding label.
        """
        if self.model_type == "cnn":
            return self.data[index].unsqueeze(0), self.labels[index]
        if self.model_type == "seq":
            return self.data[index].unsqueeze(-1), self.labels[index]
        if self.model_type == "mlp":
            return self.data[index], self.labels[index] 

        

    def get_class_mapping(self) -> Dict[int, str]:
        """
        Returns the mapping of class indices to class names.

        Returns:
            Dict[int, str]: Mapping of class indices to class names.
        """
        return {i: class_name for i, class_name in enumerate(self.label_encoder.classes_)}



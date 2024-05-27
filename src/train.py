# -*- coding: utf-8 -*-
"""
Developed by: Daniel Crovo y Sebastián Franco
Tesis: Brain Structures Classification from iEEG signals 


"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch 
import multiprocessing #
from asp_model import UNET # .py previamente desarrollado
import torch.optim as optim # para optimización de parámetros
import torch.nn as nn # para crear, definir y personalizar diferentes tipos de capas, modelos y criterios de pérdida en DL
from tqdm import tqdm # para agregar barras de progreso a bucles o iteraciones de larga duración
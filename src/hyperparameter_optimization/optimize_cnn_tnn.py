import os
import sys
sys.path.append('../')

from conv_transformer import ConvTransformerModel

from utils import get_loaders, import_checkpoint, save_checkpoint
import torch
import multiprocessing
import mlflow
import mlflow.pytorch
import torch
import torch.optim as optim
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix,roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import psutil
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pprint import pformat
import optuna

## MLFLOW Setup
os.environ['AWS_ACCESS_KEY_ID'] = 'dIgexhE2iDrGls2qargL'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'IzEzgQpztotDnrIInJdUfUIYngpjJoT18d0FDZf7'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://localhost:9000"
os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
mlflow.set_tracking_uri("http://localhost:5000")
print('tracking uri:', mlflow.get_tracking_uri())


# Configuration
DATA_DIR = '../../data/data_normalized_exp2'
SEQ_LENGTH = 300
NUM_EPOCHS = 15
EXPERIMENT_NAME = "CNN_TNN_OPTIMIZATION"
PIN_MEMORY = True
LOAD_MODEL = False
NUM_WORKERS = multiprocessing.cpu_count()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = SEQ_LENGTH
NUM_CLASSES = 4
CHECKPOINTS_PATH = '../../models/checkpoints'
DURATION = 7*3600  # Set the timeout duration in seconds (e.g., 3600 seconds = 1 hour)
N_TRIALS =100


def get_model_size(model):
    param_size = 0
    buffer_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb


def train_model(model: nn.Module, train_loader: DataLoader, optimizer: optim.Optimizer, 
                criterion: nn.Module, num_epochs: int, device: torch.device, epoch, trial):
    model.train()
    scaler = GradScaler()  # For mixed precision training
    running_loss = 0.0
    y_true_train = []
    y_pred_train = []

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Trial {trial.number}", unit="batch")

    for batch_idx, (inputs, labels) in enumerate(progress_bar):
        inputs, labels = inputs.to(device), labels.to(device)

        with autocast():  # Mixed precision training
            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze())

        scaler.scale(loss).backward()  # Backpropagation
        scaler.step(optimizer)  # Update weights
        scaler.update()
        optimizer.zero_grad()  # Reset gradients after updating weights

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        y_true_train.extend(labels.squeeze().cpu().numpy())
        y_pred_train.extend(predicted.cpu().numpy())

        progress_bar.set_postfix(train_loss=loss.item())

    avg_loss = running_loss / len(train_loader)
    train_f1 = precision_recall_fscore_support(y_true_train, y_pred_train, average='weighted', zero_division=0)[2]
    torch.cuda.empty_cache()
    return avg_loss, train_f1


def validate_model(model: nn.Module, val_loader: DataLoader, 
                   device: torch.device, criterion, trial):
    model.eval()
    val_loss = 0    
    y_true_val = []
    y_pred_val = []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Validating - Trial {trial.number}", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.squeeze())

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

            y_true_val.extend(labels.squeeze().cpu().numpy())
            y_pred_val.extend(predicted.cpu().numpy())

            torch.cuda.empty_cache()
    avg_val_loss = val_loss / len(val_loader)
    test_accuracy = accuracy_score(y_true_val, y_pred_val)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_val, y_pred_val, average='weighted', zero_division=0)

    print(f'Accuracy of the model on the val data: {test_accuracy:.2f}%')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

    return avg_val_loss, f1

def suggest_hyperparameters(trial):
    num_heads_choices = [2, 4, 8]
    transformer_dim_choices = [64, 128, 256, 512]
    num_heads = trial.suggest_categorical("num_heads", num_heads_choices)
    transformer_dim = trial.suggest_categorical("transformer_dim", transformer_dim_choices)
    lr = trial.suggest_float("lr", 1e-6, 1e-3, log=True)
    dropout = trial.suggest_float("dropout", 0.0, 0.7, step=0.1)
    batch_size = trial.suggest_int("batch_size", 32, 200)
    transformer_depth = trial.suggest_int('transformer_depth', 1, 7)
    fc_neurons1 = trial.suggest_int('fc_neurons1', 128, 512)
    fc_neurons2 = trial.suggest_int('fc_neurons2', 64, 256)
    fc_transformer = trial.suggest_int('fc_transformer', 64, 512)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    activation_choice = trial.suggest_categorical("activation", ["ReLU", "GELU"])

    print(f"Suggested hyperparameters: \n{pformat(trial.params)}")
    return lr, dropout, batch_size, transformer_dim, num_heads, transformer_depth, fc_neurons1, fc_neurons2, fc_transformer, weight_decay, activation_choice

    
def objective(trial, experiment, device, options=None):
    best_val_loss = float('inf')
    
    with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
        mlflow.log_params(options)
        lr, dropout, batch_size, transformer_dim, num_heads, transformer_depth, fc_neurons1, fc_neurons2, fc_transformer, weight_decay, activation_choice = suggest_hyperparameters(trial)
        active_run = mlflow.active_run()
        print(f"Starting run {active_run.info.run_id} and trial {trial.number}")

        mlflow.log_param("lr", lr)
        mlflow.log_param("dropout", dropout)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("transformer_dim", transformer_dim)
        mlflow.log_param("num_heads", num_heads)
        mlflow.log_param("transformer_depth", transformer_depth)
        mlflow.log_param("fc_neurons1", fc_neurons1)
        mlflow.log_param("fc_neurons2", fc_neurons2)
        mlflow.log_param("fc_transformer", fc_transformer)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("activation", activation_choice)

        # Map the activation choice string to the corresponding class
        if activation_choice == "ReLU":
            activation = nn.ReLU()
        else:
            activation = nn.GELU()

        # Model parameters
        input_size = SEQ_LENGTH
        num_classes = 4
        train_loader, val_loader, _, _ = get_loaders(data_dir=DATA_DIR, 
                                                     with_val_loader=True, 
                                                     batch_size=batch_size, 
                                                     num_workers=NUM_WORKERS,
                                                     pin_memory=PIN_MEMORY, 
                                                     test_size=0.15, 
                                                     seq_length=SEQ_LENGTH, 
                                                     model_type="cnn")

        model = ConvTransformerModel(
            input_size=input_size,
            num_classes=num_classes,
            transformer_dim=transformer_dim,
            num_heads=num_heads,
            transformer_depth=transformer_depth,
            fc_neurons=[fc_neurons1, fc_neurons2],
            fc_transformer=fc_transformer,
            dropout=dropout,
            activation=activation
        ).to(DEVICE)
        
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
        criterion = nn.CrossEntropyLoss().to(device)

        for epoch in range(0, options["epochs"]):
            avg_train_loss, train_f1 = train_model(model, train_loader, optimizer, criterion, options["epochs"], device, epoch, trial)
            avg_val_loss, val_f1 = validate_model(model, val_loader, device, criterion, trial)

            scheduler.step(avg_val_loss)

            if avg_val_loss <= best_val_loss:
                best_val_loss = avg_val_loss

            trial.report(avg_val_loss, step=epoch)

            if trial.should_prune():
                raise optuna.TrialPruned()
            
            mlflow.log_metric("avg_train_loss", avg_train_loss, step=epoch)
            mlflow.log_metric("train_f1", train_f1, step=epoch)
            mlflow.log_metric("avg_val_loss", avg_val_loss, step=epoch)
            mlflow.log_metric("val_f1", val_f1, step=epoch)

            print(f"Epoch {epoch+1}/{options['epochs']} - Train Loss: {avg_train_loss:.4f} - Train F1: {train_f1:.4f} - Val Loss: {avg_val_loss:.4f} - Val F1: {val_f1:.4f}")

    return best_val_loss

            

def main():
    options = {
        "experiment_name": EXPERIMENT_NAME,
        "epochs": NUM_EPOCHS,
        "save_model": True
    }

    # Create mlflow experiment if it doesn't exist already
    experiment_name = options["experiment_name"]
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
        experiment = mlflow.get_experiment_by_name(experiment_name)
    mlflow.set_experiment(experiment_name)

    optuna.logging.set_verbosity(optuna.logging.INFO)
    # Create the optuna study which shares the experiment name
    study = optuna.create_study(study_name=experiment_name, direction="minimize")
    study.optimize(lambda trial: objective(trial, experiment, DEVICE, options), n_trials=N_TRIALS, timeout=DURATION, show_progress_bar=True)

    # Filter optuna trials by state
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("\n++++++++++++++++++++++++++++++++++\n")
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Trial number: ", trial.number)
    print("  Value (Val Loss): ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Log the best model
    if options["save_model"]:
        best_model_params = trial.params
        best_model = ConvTransformerModel(
            input_size=SEQ_LENGTH,
            num_classes=NUM_CLASSES,
            transformer_dim=best_model_params['transformer_dim'],
            num_heads=best_model_params['num_heads'],
            transformer_depth=best_model_params['transformer_depth'],
            fc_neurons=[best_model_params['fc_neurons1'], best_model_params['fc_neurons2']],
            fc_transformer=best_model_params['fc_transformer'],
            dropout=best_model_params['dropout'],
            activation=nn.ReLU()
        ).to(DEVICE)
        

        model_path = os.path.join(CHECKPOINTS_PATH, f"best_model_trial_{trial.number}.pth")
        torch.save(best_model.state_dict(), model_path)
        print(f"Best model saved as {model_path}")

        mlflow.pytorch.log_model(best_model, "best_model", registered_model_name='best_cnn_tnn_model')
        # mlflow.log_artifact(model_path, artifact_path="models")
        print(f"Best model saved as {model_path}")
        # Get the artifact URI dynamically





if __name__ == '__main__':
    main()
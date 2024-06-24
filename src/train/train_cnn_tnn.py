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
import math


## MLFLOW Setupu
os.environ['AWS_ACCESS_KEY_ID'] = 'dIgexhE2iDrGls2qargL'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'IzEzgQpztotDnrIInJdUfUIYngpjJoT18d0FDZf7'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
mlflow.set_tracking_uri("http://localhost:5000")
print('tracking uri:', mlflow.get_tracking_uri())

# Configuration
DATA_DIR = '../../data/data_normalized_exp2'
SEQ_LENGTH = 500
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.0001
EXPERIMENT_NAME = "OPTIMIZATION_CNN_TNN"
PIN_MEMORY = True
LOAD_MODEL = False
NUM_WORKERS = multiprocessing.cpu_count()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_SIZE = SEQ_LENGTH
NUM_CLASSES = 4
CHECKPOINTS_PATH = '../../models/checkpoints'


def get_model_size(model):
    param_size = 0
    buffer_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: optim.Optimizer, 
                criterion: nn.Module, num_epochs: int, device: torch.device, save_checkpoint_interval: int = 10, 
                early_stopping_patience: int = 15, checkpoint_dir: str = '../models/checkpoints', 
                results_dir: str = '../models/results', accumulation_steps: int = 2,
                cnn=False, model_name='CNN'):
    """
    Train a deep learning model with the given parameters and log metrics to MLflow.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        optimizer (optim.Optimizer): Optimizer for updating model parameters.
        criterion (nn.Module): Loss function.
        num_epochs (int): Number of epochs to train.
        device (torch.device): Device to use for training (CPU or GPU).
        save_checkpoint_interval (int, optional): Interval for saving checkpoints. Default is 10.
        early_stopping_patience (int, optional): Patience for early stopping. Default is 15.
        checkpoint_dir (str, optional): Directory to save checkpoints. Default is 'checkpoints'.
        results_dir (str, optional): Directory to save results. Default is 'results'.
        accumulation_steps (int, optional): Number of steps to accumulate gradients before updating weights. Default is 2.
        cnn (bool, optional): If True, use CNN mode. Default is False.
        model_name (str, optional): Name of the model for saving checkpoints. Default is 'CNN'.
    """
    scaler = GradScaler()  # For mixed precision training
    best_val_loss = float('inf')  # Track the best validation loss for early stopping
    patience_counter = 0  # Counter for early stopping

    # Ensure results and checkpoint directories exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    train_metrics = []
    val_metrics = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        y_true_train = []
        y_pred_train = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        optimizer.zero_grad()  # Reset gradients at the start of each epoch

        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs, labels = inputs.to(device), labels.to(device)

            with autocast():  # Mixed precision training
                if not cnn:
                    outputs = model(inputs)
                else:
                    outputs, _ = model(inputs)
                    

                loss = criterion(outputs, labels.squeeze())

            scaler.scale(loss).backward()  # Backpropagation

            scaler.step(optimizer)  # Update weights
            scaler.update()
            optimizer.zero_grad()  # Reset gradients after updating weights

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            y_true_train.extend(labels.squeeze().cpu().numpy())
            y_pred_train.extend(predicted.cpu().numpy())

            avg_loss = running_loss / (batch_idx + 1)
            train_accuracy = accuracy_score(y_true_train, y_pred_train)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true_train, y_pred_train, average='weighted', zero_division=0)

            progress_bar.set_postfix(train_loss=avg_loss, train_accuracy=train_accuracy, train_precision=precision, train_recall=recall, train_f1=f1)

        # Log training metrics to MLflow
        mlflow.log_metric("train_loss", avg_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
        mlflow.log_metric("train_precision", precision, step=epoch)
        mlflow.log_metric("train_recall", recall, step=epoch)
        mlflow.log_metric("train_f1_score", f1, step=epoch)

        # Store training metrics in DataFrame
        train_metrics.append({
            "epoch": epoch + 1,
            "model_name": model_name,
            "train_loss": avg_loss,
            "train_accuracy": train_accuracy,
            "train_precision": precision,
            "train_recall": recall,
            "train_f1": f1
        })

        # Validation step
        model.eval()
        val_loss = 0.0
        y_true_val = []
        y_pred_val = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                with autocast():  # Mixed precision inference
                    if not cnn:
                        outputs = model(inputs)
                    else:
                        outputs, _ = model(inputs)
                    loss = criterion(outputs, labels.squeeze())

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                y_true_val.extend(labels.squeeze().cpu().numpy())
                y_pred_val.extend(predicted.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(y_true_val, y_pred_val)
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(y_true_val, y_pred_val, average='weighted', zero_division=0)

        # Log validation metrics to MLflow
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
        mlflow.log_metric("val_precision", val_precision, step=epoch)
        mlflow.log_metric("val_recall", val_recall, step=epoch)
        mlflow.log_metric("val_f1_score", val_f1, step=epoch)

        # Store validation metrics in DataFrame
        val_metrics.append({
            "epoch": epoch + 1,
            "model_name": model_name,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_f1": val_f1
        })

        # Update the progress bar with validation metrics
        progress_bar.set_postfix(train_loss=avg_loss, train_accuracy=train_accuracy, train_precision=precision, train_recall=recall, train_f1=f1, val_loss=avg_val_loss, val_accuracy=val_accuracy, val_precision=val_precision, val_recall=val_recall, val_f1=val_f1)

        # Save checkpoint every 'save_checkpoint_interval' epochs
        if (epoch + 1) % save_checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{model_name}.pth.tar')
            save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint_path)
            mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")

        # Early stopping based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0  # Reset counter if we get a new best validation loss
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss.")
            mlflow.log_param(f"{model_name}_epochs_actual", epoch + 1)
            break

        # Clear CUDA cache after each epoch
        torch.cuda.empty_cache()

    # Save training and validation metrics as CSV files
    train_metrics_df = pd.DataFrame(train_metrics)
    val_metrics_df = pd.DataFrame(val_metrics)
    train_metrics_df.to_csv(os.path.join(results_dir, f'train_metrics_{model_name}.csv'), index=False)
    val_metrics_df.to_csv(os.path.join(results_dir, f'val_metrics_{model_name}.csv'), index=False)

    # Clear CUDA cache at the end of training
    torch.cuda.empty_cache()



def evaluate_model(model: nn.Module, test_loader: DataLoader, dataset: Dataset, 
                   device: torch.device, img_path: str, results_dir: str,
                   run_name: str, batch_size: int = 16, cnn=False):
    """
    Evaluate a deep learning model and log metrics to MLflow.

    Args:
        model (nn.Module): The model to evaluate.
        test_loader (DataLoader): DataLoader for the test data.
        dataset (Dataset): The dataset containing the test data.
        device (torch.device): Device to use for evaluation (CPU or GPU).
        img_path (str): Path to save the confusion matrix image.
        run_name (str): Name of the MLflow run.
        batch_size (int, optional): Batch size for evaluation. Default is 16.
    """
    model.eval()
    y_true_test = []
    y_pred_test = []
    feature_maps = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            if not cnn: 
                outputs = model(inputs)

            else:
                outputs, feature_map = model(inputs)
                feature_maps.append([fm.cpu() for fm in feature_map])  # Move feature maps to CPU to free GPU memory

            _, predicted = torch.max(outputs, 1)
            y_true_test.extend(labels.squeeze().cpu().numpy())
            y_pred_test.extend(predicted.cpu().numpy())

            # Clear cache to free up memory
            torch.cuda.empty_cache()

    test_accuracy = accuracy_score(y_true_test, y_pred_test)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_test, y_pred_test, average='weighted', zero_division=0)

    print(f'Accuracy of the model on the test data: {test_accuracy:.2f}%')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_precision", precision)
    mlflow.log_metric("test_recall", recall)
    mlflow.log_metric("test_f1", f1)

    # Confusion matrix
    cm = confusion_matrix(y_true_test, y_pred_test)
    cm_df = pd.DataFrame(cm, index=dataset.label_encoder.classes_, columns=dataset.label_encoder.classes_)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    img_file = os.path.join(img_path, f"confusion_matrix_{run_name}.png")
    cm_file = os.path.join(results_dir, f"confusion_matrix_{run_name}.csv")

    plt.savefig(img_file)
    cm_df.to_csv(cm_file)

    mlflow.log_artifact(img_file)
    mlflow.log_artifact(cm_file)

    plt.close()

    # Save feature maps and labels to file
    if cnn:
        feature_maps_file = os.path.join(img_path, f"feature_maps_{run_name}.pt")
        torch.save((feature_maps, y_true_test, y_pred_test), feature_maps_file)
    # mlflow.log_artifact(feature_maps_file, artifact_path="feature_maps")

    return y_true_test, y_pred_test


def main():
    train_loader, val_loader, test_loader, dataset = get_loaders(data_dir=DATA_DIR, 
                                                                 with_val_loader=True, 
                                                                 batch_size=BATCH_SIZE, 
                                                                 num_workers=NUM_WORKERS,
                                                                 pin_memory=PIN_MEMORY, 
                                                                 test_size=0.15, 
                                                                 seq_length=SEQ_LENGTH, 
                                                                 model_type="cnn")
    
        # Model parameters
    input_size = SEQ_LENGTH  # Use the sequence length provided by your dataset
    num_classes = 4  # Number of classes for classification
    conv_filters = [64, 128]  # Reduced number of filters to save memory
    transformer_dim = 128  # Smaller transformer dimension
    num_heads = 4  # Fewer attention heads
    transformer_depth = 2  # Fewer transformer layers
    fc_neurons = [512, 128]  # Reduced fully connected layer sizes
    dropout = 0.3  # Dropout rate

    model = ConvTransformerModel(
        input_size=input_size,
        num_classes=num_classes,
        transformer_dim=transformer_dim,
        num_heads=num_heads,
        transformer_depth=transformer_depth,
        fc_neurons=fc_neurons,
        dropout=dropout,
        activation=nn.ReLU()
    ).to(DEVICE)
    print(f'Model size: {get_model_size(model):.3f} MB')
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    criterion =  nn.CrossEntropyLoss()  
    mlflow.set_experiment(EXPERIMENT_NAME)
    # Train and Evaluate the Model with MLflow
    run_name = "run_CNN_TNN_optimize"
    model_name = "CNN_TNN_optimized"
    results_dir = "../../models/results"
    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_param("epochs", NUM_EPOCHS)
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("model", model_name)
        mlflow.log_param("input_size", SEQ_LENGTH)
        mlflow.log_param("num_classes", NUM_CLASSES)
        mlflow.log_dict(dataset.get_class_mapping(), "class_mapping.json")

        # Train and Evaluate the Model
        train_model(model, train_loader,val_loader, optimizer, criterion, NUM_EPOCHS, DEVICE, 
                    save_checkpoint_interval=10, checkpoint_dir=CHECKPOINTS_PATH, 
                    model_name=model_name, early_stopping_patience=10, cnn=False)
        _,_ = evaluate_model(model, test_loader, dataset, DEVICE, 
                            results_dir=results_dir,
                            img_path='../../plots', 
                            run_name=run_name,
                            cnn=False)

        # Log the model
        mlflow.pytorch.log_model(model, model_name)

if __name__ == '__main__':
    main()


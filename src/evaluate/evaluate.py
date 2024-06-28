import os
import sys
sys.path.append('../')

import os
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from ieeg_dataset import IeegDataset  
import json


## MLFLOW Setup
os.environ['AWS_ACCESS_KEY_ID'] = 'dIgexhE2iDrGls2qargL'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'IzEzgQpztotDnrIInJdUfUIYngpjJoT18d0FDZf7'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000'
os.environ['MLFLOW_S3_IGNORE_TLS'] = 'true'
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
mlflow.set_tracking_uri("http://localhost:5000")
print('tracking uri:', mlflow.get_tracking_uri())

# Configuration
DATA_DIR = '../../data/test'  # Directory containing the test data
MODEL_NAME = 'CNN_TNN_optimized'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
RESULTS_DIR = './results'
PLOTS_DIR = './plots'
VERSION = 2
# Ensure results and plots directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Set up MLflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("FINAL_CNN_TNN_TRAINING")



model_uri = f"models:/{MODEL_NAME}/{VERSION}"
model = mlflow.pyfunc.load_model(model_uri=model_uri)
# Load the test data
test_dataset = IeegDataset(data_dir=DATA_DIR, seq_length=300, model_type='cnn')
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

def evaluate_model(model, test_loader, device, class_mapping, results_dir, plots_dir):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=[class_mapping[i] for i in range(len(class_mapping))], 
                         columns=[class_mapping[i] for i in range(len(class_mapping))])

    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    metrics_path = os.path.join(results_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)

    # Log metrics

    # Plot and save confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    cm_plot_path = os.path.join(plots_dir, 'confusion_matrix.png')
    plt.savefig(cm_plot_path)

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

if __name__ == "__main__":
    # Get class mapping
    class_mapping = test_dataset.get_class_mapping()

    # Evaluate the model
    evaluate_model(model, test_loader, DEVICE, class_mapping, RESULTS_DIR, PLOTS_DIR)

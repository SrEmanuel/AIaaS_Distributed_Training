import datetime
import time
import json
from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
import requests
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
from typing import Optional
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import warnings
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import torchvision.models as models
from torchvision import transforms
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.datasets import ImageFolder
import time
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os
import seaborn as sns

START = time.time()
DATE_NOW = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IAAS_ENDPOINT_LOCAL="http://localhost:8080/api/v1/training"
IAAS_ENDPOINT="https://api.iaas.emanuelm.dev/api/v1/training"

parser = argparse.ArgumentParser(description='Distbelief training example')
parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--port', type=str, default='3002')
parser.add_argument('--model_name', type=str, help='Give the model name')
parser.add_argument('--trainingUuid', type=str, help='Training UUID')
parser.add_argument('--numRounds', type=str, help='Number of rounds')

args = parser.parse_args()

if(args.trainingUuid is None):
    raise Exception("You must inform the training UUID!!")

def save_confusion_matrix(y_true, y_pred, class_names, output_dir, accuracy, loss, elapsed_time, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.2%}, Loss: {loss:.4f}, Time: {elapsed_time:.2f} seconds')

    # Save the plot as a PDF
    output_dir = output_dir + r'/outputs/' + model_name
    os.makedirs(output_dir, exist_ok=True)
    result_dir = output_dir + '/' + model_name +'_' + DATE_NOW
    os.makedirs(result_dir, exist_ok=True)
    output_path = os.path.join(result_dir, f'{model_name}_confusion_matrix.pdf')
    plt.savefig(output_path, format="pdf", bbox_inches='tight')

    # Calculate precision, recall, and F1-score
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    metrics_path = os.path.join(result_dir, "metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"Accuracy: {accuracy:.2%}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
        f.write(f"Loss: {loss:.4f}\n")
        f.write(f"Elapsed Time: {elapsed_time:.2f} seconds")

class Net(nn.Module):

    def __init__(self, model_name, class_num, output) -> None:
        super(Net, self).__init__()

        if model_name == "alexnet":
            self.model = models.alexnet(weights='DEFAULT')
            num_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_features, output)
        elif model_name == "resnet":
            self.model = models.resnet50(weights='DEFAULT')
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, output)
        else:
            raise ValueError(f"Modelo não suportado: {model_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def test(net, testloader, output_dir):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    true_labels = []
    predicted_labels = []
    y_true = []
    y_pred = []
    net.eval()

    with torch.no_grad():
        for inputs, labels in tqdm(testloader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)

            loss += criterion(outputs, labels).item() * inputs.size(0)

            correct += (predicted == labels).sum().item()

            # correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()

            # Collect true and predicted labels for confusion matrix
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            y_true += labels.tolist()
            y_pred += predicted.tolist()

    end_time = time.time()
    elapsed_time = end_time - START
    accuracy = correct / len(testloader.dataset)
    real_loss = loss / len(testloader.dataset)

    print(f"Test Loss: {real_loss:.4f}, Test Accuracy: {accuracy:.2%}")

    # Save the confusion matrix and accuracy
    class_names = ["benign", "malignant"]
    # save_confusion_matrix(true_labels, predicted_labels, class_names, output_dir, accuracy, real_loss, elapsed_time)
    save_confusion_matrix(y_true, y_pred, class_names, output_dir, accuracy, real_loss, elapsed_time, args.model_name)

    return real_loss, accuracy

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    print("Metrics: "+str(metrics))
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    print("Acuracies list: "+str(len(accuracies)))
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    print(f"\n### Server-side evaluation accuracy: "+str(float(sum(accuracies) / sum(examples))))
    return {"accuracy": sum(accuracies) / sum(examples)}


def load_data():
    # Load the breast cancer dataset (modify the paths accordingly)
    input_size = 224
    data_transforms = {
        'transform': transforms.Compose([
            transforms.Resize([input_size, input_size], antialias=True),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    trainset = ImageFolder("./dataset/production/train", transform=data_transforms['transform'])
    testset = ImageFolder("./dataset/production/test", transform=data_transforms['transform'])
    return DataLoader(trainset, batch_size=16, shuffle=True), DataLoader(testset)




_, testloader = load_data()



# The `evaluate` function will be by Flower called after every round
def evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    net = Net(model_name=args.model_name, class_num=2, output=2).to(DEVICE)
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy = test(net, testloader, output_dir=".")
    torch.save(net.state_dict(), 'server_model_aggregated.pth')
    accuracy_percent = accuracy * 100  # Multiplica a precisão por 100 para obter o valor percentual
    print(f"\n### Server-side evaluation loss {loss} / accuracy {accuracy_percent:.2f}% ###\n")
    return loss, {"accuracy": accuracy}



# Define strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    evaluate_fn=evaluate,
    evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
)


def performTrainingRequest(status: str):
    
    file = open("./scripts/teste.pdf", 'rb')
    
    training_finish_dto = {
        "timestamp": time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(time.time())),
        "status": status,
        "data": "Training data",
        "errors": []
    }
    
    parts = {
        'file': ('teste.pdf', file),
        'request': ('json', json.dumps(training_finish_dto), 'application/json')
    }

    
    requests.post(IAAS_ENDPOINT_LOCAL+"/finishTraining/"+args.trainingUuid, files=parts)

net = Net(model_name=args.model_name, class_num=2, output=2).to(DEVICE)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:"+args.port,
    config=fl.server.ServerConfig(num_rounds=int(args.numRounds)),
    strategy=strategy
)


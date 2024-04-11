from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
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


warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Distbelief training example')
parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--port', type=str, default='3002')
parser.add_argument('--world_size', type=int)
parser.add_argument('--rank', type=int)
parser.add_argument('--model_name', help='Give the model name')
parser.add_argument('--dataset', type=str, help='Nome do diretório do dataset')
parser.add_argument("--epochs", type=int)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--dataset_id", type=int, help='ID do DataSet')
parser.add_argument("--batch_size", type=int, default=32, help='Batch Size do Dataset')
args = parser.parse_args()

class Net(nn.Module):

    def __init__(self, model_name, class_num, output) -> None:
        super(Net, self).__init__()

        if model_name == "alexnet":
            self.model = models.alexnet(weights='DEFAULT')
            num_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_features, 2)
        elif model_name == "resnet":
            self.model = models.resnet50(weights='DEFAULT')
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, 2)
        else:
            raise ValueError(f"Modelo não suportado: {model_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

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

#_, testloader = create_federated_testloader()
net = Net(model_name=args.model_name, class_num="", output="").to(DEVICE)

# The `evaluate` function will be by Flower called after every round
def evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    net = LSTMModel(input_size=49, hidden_size=32, num_layers=5, output_size=2).to(DEVICE)
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, accuracy = test(net, testloader)
    torch.save(net.state_dict(), 'server_model_aggregated.pth')
    accuracy_percent = accuracy * 100  # Multiplica a precisão por 100 para obter o valor percentual
    print(f"\n### Server-side evaluation loss {loss} / accuracy {accuracy_percent:.2f}% ###\n")
    return loss, {"accuracy": accuracy}



# Define strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
    #evaluate_fn=evaluate,
    evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
)



# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy
)
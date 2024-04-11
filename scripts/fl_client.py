import warnings
from collections import OrderedDict
import numpy as np
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
import time
import datetime
import random
import torchvision.models as models
from torchvision import transforms
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.datasets import ImageFolder

# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
START = time.time()
DATE_NOW = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


parser = argparse.ArgumentParser(description='Distbelief training example')
parser.add_argument('--ip', type=str, default='127.0.0.1')
parser.add_argument('--port', type=str, default='3002')
parser.add_argument('--world_size', type=int)
parser.add_argument('--rank', type=int)
parser.add_argument('--model_name', type=str, help='Give the model name')
parser.add_argument('--dataset', type=str, help='Nome do diretório do dataset')
parser.add_argument("--epochs", type=int)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--dataset_id", type=int, help='ID do DataSet')
parser.add_argument("--batch_size", type=int, default=32, help='Batch Size do Dataset')
parser.add_argument("--optim", type=str, help='Optimizer to choose: Adam or SGD')
args = parser.parse_args()

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


def train(net, trainloader, epochs, output_dir, model_name):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # Lists to store loss and accuracy per epoch
    train_loss = []
    train_accuracy = []
    preds = 0
    net.train()
    for epoch in range(epochs):
        correct, total, total_loss = 0, 0, 0.0
        for inputs, labels in tqdm(trainloader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            net.to(DEVICE)

            outputs = net(inputs)

            optimizer.zero_grad()

            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            total += labels.size(0)
            correct += torch.sum(preds == labels.data)
            # correct += (torch.max(outputs.data, 1)[1] == labels.to(DEVICE)).sum().item()

        # Calculate accuracy and save loss and accuracy
        accuracy = correct.double() / len(trainloader.dataset)
        epoch_loss = total_loss / len(trainloader.dataset)
        # accuracy = correct / total
        train_loss.append(epoch_loss)
        train_accuracy.append(accuracy)

        print(f"Epoch {epoch + 1}/{epochs}: Loss = {epoch_loss:.4f}, Accuracy = {accuracy:.2%}")

    # Create and save loss and accuracy plots
    epochs_range = np.arange(1, epochs + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_loss, 'b', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    # plt.plot(epochs_range, train_accuracy, 'r', label='Training Accuracy')
    plt.plot(epochs_range, [acc.cpu().numpy() for acc in train_accuracy], 'r', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()

    # Save the plot as a PDF
    output_dir = output_dir + r'/outputs/' + model_name
    os.makedirs(output_dir, exist_ok=True)
    result_dir = output_dir + '/' + model_name + '_' + DATE_NOW
    os.makedirs(result_dir, exist_ok=True)
    output_path = os.path.join(result_dir, f'{model_name}_loss_accuracy_plots.pdf')
    plt.savefig(output_path, format="pdf", bbox_inches='tight')


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
    save_confusion_matrix(y_true, y_pred, class_names, output_dir, accuracy, real_loss, elapsed_time)

    return real_loss, accuracy


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

    trainset = ImageFolder("../dataset/production/train", transform=data_transforms['transform'])
    testset = ImageFolder("../dataset/production/test", transform=data_transforms['transform'])
    return DataLoader(trainset, batch_size=16, shuffle=True), DataLoader(testset)




trainloader, testloader = load_data()
if torch.cuda.is_available():
    print("Placa de Video")
else:
    print("CPU")

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=args.epochs, model_name=net)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        print("Acurácia do Cliente: "+str(args.dataset_id)+str(" eh: ")+str(accuracy))
        return float(loss), len(testloader.dataset), {"accuracy": round(float(accuracy) * 100, 2)}


net = Net(model_name=args.model_name, class_num=2, output=2).to(DEVICE)
# Start Flower client
fl.client.start_numpy_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)
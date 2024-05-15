#####################
# Install libraries #
#####################

import os

os.system("pip install opencv-python transformers")

import pickle
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
from typing import Tuple, List, Dict, Any

import torch
import torch.nn as nn
from torch.nn import LSTM, Sequential, Linear, Sigmoid, ReLU, Conv2d
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import vgg16, resnet50

from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit, StratifiedGroupKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, roc_curve, auc

from transformers import AutoImageProcessor, AutoModelForImageClassification

categories = {'Neutral': 0, 'Confusion': 1}
AMOUNT_OF_FRAMES = 40
SKIP_FRAMES = 1

class GIFDataset(Dataset):
    def __init__(self, ids, labels, transform, processor=None, skipframes: int=1):      
        self.ids = ids
        self.labels = labels
        self.skipframes = skipframes
        self.transform = transform
        self.processor = processor
        self.labels_dict = categories
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        relevant_landmark_ids = [61, 292, 0, 17, 50, 280, 48, 4, 289, 206, 426, 133, 130, 159, 145, 362, 359, 386, 374, 122, 351, 46, 105, 107, 276, 334, 336]
        
        frames_tr = pd.read_csv(self.ids[idx])
        frames_tr = frames_tr.iloc[:AMOUNT_OF_FRAMES // self.skipframes:self.skipframes, :]
        frames_tr = frames_tr.drop(frames_tr.columns[2::3], axis=1)

        # relevant_columns = [2 * idx for idx in relevant_landmark_ids] + [2 * idx + 1 for idx in relevant_landmark_ids]
        # frames_tr = frames_tr.iloc[:, relevant_columns]
        
        frames_tr = torch.tensor(frames_tr.to_numpy(), dtype=torch.float32)
        frames_tr = normalize_frames(frames_tr)
        
        while len(frames_tr) < AMOUNT_OF_FRAMES // self.skipframes:
            last_row = frames_tr[-1].unsqueeze(0)
            frames_tr = torch.cat((frames_tr, last_row), dim=0)
        
        label = float(self.labels_dict[self.labels[idx]])
        # print(self.labels[idx], label, len(frames_tr))

        # frames_tr_even = frames_tr[:, ::2]
        # frames_tr_odd = frames_tr[:, 1::2]
        
        # frames_tr = torch.stack((frames_tr_even, frames_tr_odd))
        # frames_tr.view(2, AMOUNT_OF_FRAMES // SKIP_FRAMES, 468)
        
        return frames_tr, label

def normalize_frames(frames_tr):
    norm_frames_tr = torch.zeros(frames_tr.shape)
    for index, frame in enumerate(frames_tr):
        frame_x = frame[::2]
        frame_y = frame[1::2]
        max_x = max(frame[::2])
        min_x = min(frame[::2])
        max_y = max(frame[1::2])
        min_y = min(frame[1::2])

        # plt.figure(figsize=(6, 6))
        # plt.scatter(frame_x, frame_y, c='r', marker='o', s=1)
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.show()

        frame_x = (frame_x - min_x) / (max_x - min_x)
        frame_y = (frame_y - min_y) / (max_y - min_y)
        # plt.figure(figsize=(6, 6))
        # plt.scatter(frame_x, frame_y, c='r', marker='o', s=1)
        # plt.xlim(0, 1)
        # plt.ylim(0, 1)
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.show()

        norm_frames_tr[index][::2] = frame_x
        norm_frames_tr[index][1::2] = frame_y

    return norm_frames_tr

def get_vids(path2folder: str, maxpercat=None) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Loop over a folder and put the data into lists.

    :param path2folder: Path to folder, has to have 2 subfolders "NO" and "YES".
    :return ids: List of names of paths.
    :return labels: List of NO and YES.
    :return groups: List of prefixes referring to the date of the experiment.
    :return listOfCats: ["Neutral", "Confusion"]
    """
    listOfCats = ["Neutral", "Confusion"]
    ids = []
    labels = []
    groups = []
    for catg in listOfCats:
        path2catg = os.path.join(path2folder, catg)
        if maxpercat:
            if catg == "Confusion":
                listOfSubCats = os.listdir(path2catg)[:maxpercat[0]]
            else:
                listOfSubCats = os.listdir(path2catg)[:maxpercat[1]]
        else:
            listOfSubCats = os.listdir(path2catg)
        path2subCats = []
        for los in listOfSubCats:
            if ".ipynb_checkpoints" not in los:
                path2subCats.append(os.path.join(path2catg,los))
                groups.append(los[:19])
        ids.extend(path2subCats)
        labels.extend([catg]*len(path2subCats))
    return ids, labels, groups, listOfCats 


def get_class_percentages(ids: List[str]) -> Tuple[float, float]:
    """
    Get the percentages of the different classes.

    :param ids: The path names.
    :return YES_percentage_dataset: Percentage of the YES class.
    :return NO_percentage_dataset: Percentage of the NO class.
    """
    YES_count = 0
    NO_count = 0
    for id in ids:
        if "Neutral" in id:
            YES_count += 1
        elif "Confusion" in id:
            NO_count += 1
        else:
            print("Something went wrong!")
    YES_percentage_dataset = YES_count / len(ids)
    NO_percentage_dataset = NO_count / len(ids)
    return YES_percentage_dataset, NO_percentage_dataset


def split_dataset(ids: List[str], labels: List[str], groups: List[str], endpath: str="split_keypoint", skipframes: int=1, name: str="", is_hf: bool=False) -> None:
    """
    Split the dataset in a train, val and test dataset in a grouped and stratified approach.

    :param ids: List of ids.
    :param labels: List of labels.
    :param groups: List of groups.
    :param endpath: Name of the path where to dataset is located.
    :param skipframes: Amount of frames skipped in each video.
    :param name: Name of the type of dataset.
    """
    print("Name\t\t ids labels groups")
    print("-"*38)
    # The data has a grouped structured but should also be stratified 
    # because the NO class is underrepresented compared to the YES class
    # First group the data and check afterwards if the stratified requirement is fulfilled


    sss_train = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=42)
    train_indx, val_indx = next(sss_train.split(ids, labels))
    
    train_ids = [ids[ind] for ind in train_indx]
    train_labels = [labels[ind] for ind in train_indx]
    train_groups = [groups[ind] for ind in train_indx]
    print("Train: \t\t", len(train_ids), len(train_labels), len(Counter(train_groups)))
    
    val_ids = [ids[ind] for ind in val_indx]
    val_labels = [labels[ind] for ind in val_indx]
    val_groups = [groups[ind] for ind in val_indx]
    print("Validation: \t", len(val_ids), len(val_labels), len(Counter(val_groups)))
    
    # # Split the full dataset in a train_val and a test_set with GroupShuffleSplit
    # sss_train_val_test = GroupShuffleSplit(n_splits=2, test_size=0.1, random_state=42)
    # train_val_indx, test_indx = next(sss_train_val_test.split(ids, labels, groups))
    
    # train_val_ids = [ids[ind] for ind in train_val_indx]
    # train_val_labels = [labels[ind] for ind in train_val_indx]
    # train_val_groups = [groups[ind] for ind in train_val_indx]
    # print("Train_val: \t", len(train_val_ids), len(train_val_labels), len(Counter(train_val_groups)))
    
    # test_ids = [ids[ind] for ind in test_indx]
    # test_labels = [labels[ind] for ind in test_indx]
    # test_groups = [groups[ind] for ind in test_indx]
    # print("Test: \t\t", len(test_ids), len(test_labels), len(Counter(test_groups)))
    
    # Split the train_val and a train_set and a val_set with GroupShuffleSplit
    # sss_train_val = GroupShuffleSplit(n_splits=2, test_size=0.25, random_state=42)
    # train_indx, val_indx = next(sss_train_val.split(train_val_ids, train_val_labels, train_val_groups))
    
    # train_ids = [train_val_ids[ind] for ind in train_indx]
    # train_labels = [train_val_labels[ind] for ind in train_indx]
    # train_groups = [train_val_groups[ind] for ind in train_indx]
    # print("Train: \t\t", len(train_ids), len(train_labels), len(Counter(train_groups))) 
    
    # val_ids = [train_val_ids[ind] for ind in val_indx]
    # val_labels = [train_val_labels[ind] for ind in val_indx]
    # val_groups = [train_val_groups[ind] for ind in val_indx]
    # print("Validation: \t", len(val_ids), len(val_labels), len(Counter(val_groups)))
    print("-"*38)
    
    # Display percentages
    YES_percentage_full_dataset = labels.count("Neutral") / len(labels)
    NO_percentage_full_dataset = labels.count("Confusion") / len(labels)
    print(f"Name distribution:\t YES - NO")
    print(f"Original distribution:\t {YES_percentage_full_dataset} - {NO_percentage_full_dataset}")
    
    train_YES, train_NO = get_class_percentages(train_ids)
    val_YES, val_NO = get_class_percentages(val_ids)
    # test_YES, test_NO = get_class_percentages(test_ids)
    print(f"Train distribution:\t {train_YES} - {train_NO}")
    print(f"Val distribution:\t {val_YES} - {val_NO}")
    # print(f"Test distribution:\t {test_YES} - {test_NO}")
    print("-"*38)
    
    # Create datasets
    # https://github.com/PacktPublishing/PyTorch-Computer-Vision-Cookbook/blob/master/Chapter10/Chapter10.ipynb
    h, w = 224, 224
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_transformer = transforms.Compose([
            transforms.Resize((h,w)),
            # transforms.RandomHorizontalFlip(p=0.5),  
            # transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),    
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])
    val_transformer = transforms.Compose([
            transforms.Resize((h,w)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            ])
    test_transformer = val_transformer

    processor = None
    if is_hf:
        train_transformer, val_transformer, test_transformer = None, None, None
        processor = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")

    train_ds = GIFDataset(ids=train_ids, labels=train_labels, transform=train_transformer, processor=processor)
    val_ds = GIFDataset(ids=val_ids, labels=val_labels, transform=val_transformer, processor=processor)
    # test_ds = GIFDataset(ids=test_ids, labels=test_labels, transform=test_transformer, processor=processor)

    base_path = "/project_ghent/Master-Thesis/gif/pickled/"

    torch.save(train_ds, f"{base_path}train_ds_{endpath}_s{skipframes}_{name}.pth")
    torch.save(val_ds, f"{base_path}val_ds_{endpath}_s{skipframes}_{name}.pth")
    # torch.save(test_ds, f"{base_path}test_ds_{endpath}_s{skipframes}_{name}.pth")

    print(f"Succesfully saved at {base_path} with skipframes = {skipframes} [{name}]")


def get_dataset(endpath: str="split_keypoint", skipframes: int=1, name="") -> Tuple[GIFDataset, GIFDataset, GIFDataset]:
    """
    Retrieve the train, val and test dataset based on the specific arguments.
    
    :param endpath: Name of the path where to dataset is located.
    :param skipframes: Amount of frames skipped in each video.
    :return train_ds: The train dataset.
    :return val_ds: The validation dataset. 
    :return test_ds: The test dataset.
    """
    base_path = "/project_ghent/Master-Thesis/gif/pickled/"
    
    # Pickle dataset such that no randomisation of different machines would introduce different different results
    train_ds = torch.load(f"{base_path}train_ds_{endpath}_s{skipframes}_{name}.pth")
    val_ds = torch.load(f"{base_path}val_ds_{endpath}_s{skipframes}_{name}.pth")
    # test_ds = torch.load(f"{base_path}test_ds_{endpath}_s{skipframes}_{name}.pth")

    print(f"Succesfully retrieved at {base_path} with skipframes = {skipframes} [{name}]")
    
    return train_ds, val_ds


class BackBoneModel(nn.Module):
    """
    Class defining the backbone model
    Input: [batch_size * AMOUNT_OF_FRAMES, channels, width, height]
    Output: [batch_size * AMOUNT_OF_FRAMES, HIDDEN_SIZE]
    """
    def __init__(self, vgg: torchvision.models, hidden_size: int) -> None:
        super(BackBoneModel, self).__init__()
        self.hidden_size = hidden_size
        self.features = vgg.features
        self.avgpool = vgg.avgpool
        self.fc = Linear(512 * 7 * 7, self.hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class BackBoneModelResNet(nn.Module):
    """
    Class defining the backbone model
    Input: [batch_size * AMOUNT_OF_FRAMES, channels, width, height]
    Output: [batch_size * AMOUNT_OF_FRAMES, HIDDEN_SIZE]
    """
    def __init__(self, resnet: torchvision.models, hidden_size: int) -> None:
        super(BackBoneModelResNet, self).__init__()
        self.hidden_size = hidden_size
        # Remove the last fully connected layer
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Adjust the input feature size for the fc layer based on ResNet50 features
        self.fc = Linear(2048, self.hidden_size)  # 2048 is the feature size for ResNet50
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class BackBoneModelDima806(nn.Module):
    """
    Class defining the backbone model
    Input: [batch_size * AMOUNT_OF_FRAMES, channels, width, height]
    Output: [batch_size * AMOUNT_OF_FRAMES, HIDDEN_SIZE]
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, amount_of_frames: int=120) -> None:
        super(BackBoneModelDima806, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.amount_of_frames = amount_of_frames
        self.processor = AutoImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
        self.model = AutoModelForImageClassification.from_pretrained("dima806/facial_emotions_image_detection")
        self.model.classifier = torch.nn.Linear(in_features=768, out_features=self.input_size, bias=True)
        self.rnn = LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.relu = ReLU()
        self.fc2 = Linear(self.hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, x.shape[2], x.shape[3], x.shape[4])
        x = self.model(x).logits
        x = x.view(-1, self.amount_of_frames, self.input_size)
        x, (h, c) = self.rnn(x)
        x = x[:, -1, :]
        x = self.fc2(x)
        return x

class RecipeImageModel(nn.Module):
    """
    Class defining the RecipeImageModel model
    Input:  [batch_size, AMOUNT_OF_FRAMES, channels, width, height]
    Output: [batch_size, AMOUNT_OF_FRAMES, 1]
    """
    def __init__(self, input_size, hidden_size, num_layers=1, amount_of_frames=120, backbone="vgg") -> None:
        super(RecipeImageModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.amount_of_frames = amount_of_frames
        if backbone == "vgg":
            self.backbone = BackBoneModel(vgg16(weights="IMAGENET1K_V1"), self.input_size)
        elif backbone == "resnet":
            self.backbone = BackBoneModelResNet(resnet50(weights="IMAGENET1K_V1"), self.input_size)
        self.rnn = LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.relu = ReLU()
        self.fc2 = Linear(self.hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, x.size(2), x.size(3), x.size(4))
        x = self.backbone(x)
        x = x.view(-1, self.amount_of_frames, self.backbone.hidden_size)
        x, (h, c) = self.rnn(x)
        x = x[:, -1, :]
        x = self.fc2(x)
        return x

class GIFBlendshapeModel(nn.Module):
    """
    Class defining the classification model
    Input:  [batch_size, AMOUNT_OF_FRAMES, HIDDEN_SIZE]
    Output: [batch_size, AMOUNT_OF_FRAMES, HIDDEN_SIZE]
    """
    def __init__(self, input_size, hidden_size, num_layers=1) -> None:
        super(GIFBlendshapeModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.relu = ReLU()
        self.fc2 = Linear(self.hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, (h, c) = self.rnn(x)
        x = x[:, -1, :]
        x = self.fc2(x)
        return x

class GIFKeypointModel(nn.Module):
    """
    Class defining the classification model
    Input:  [batch_size, AMOUNT_OF_FRAMES, HIDDEN_SIZE]
    Output: [batch_size, AMOUNT_OF_FRAMES, HIDDEN_SIZE]
    """
    def __init__(self, input_size, hidden_size, num_layers) -> None:
        super(GIFKeypointModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.relu = ReLU()
        self.fc1 = Linear(2*468, input_size)
        self.fc2 = Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        # h0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size) # because batch_first
        # c0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size)
        x, (h, c) = self.rnn(x)
        x = x[:, -1, :]
        x = self.fc2(x)
        # x = self.relu(x)
        return x

def train_and_plot(model: Any, parameters: Dict[str, Any], name: str) -> None:
    """
    Train the provided models and plot the training and validation accuracy curves.
    This training loop expects a model returning logits and a BCEWithLogitsLoss optimizer.
    The training loop applies a Sigmoid() function to predict.

    :param model: The model.
    :param parameters: The parameters for the training loop.
    :param name: The name of the model.
    """
    device = parameters["device"]
    model = model.to(device)
    
    num_epochs = parameters["epochs"]
    train_loader = parameters["trainloader"]
    val_loader = parameters["valloader"]
    criterion = parameters["criterion"]
    optimizer = parameters["optimizer"]

    batch_size = parameters["batch_size"]
    amount_of_frames = parameters["amount_of_frames"]
    decision = parameters["decision"]
    
    train_accuracies = []
    val_accuracies = []
    f1_scores = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()
    
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch') as pbar:
            correct_train = 0
            total_train = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.unsqueeze(1).to(device)

                outputs = model(inputs)
                
                loss = criterion(outputs, labels.float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                pbar.update(1)
                pbar.set_postfix({'Loss': loss.item()})
                
                outputs = Sigmoid()(outputs)
                predicted_train = (outputs.data >= decision).float()

                total_train += labels.size(0)
                correct_train += (predicted_train == labels).sum().item()
    
            train_accuracy = correct_train / total_train
            train_accuracies.append(train_accuracy)

            # Empty CUDA cache after each epoch
            torch.cuda.empty_cache()
            
            # Validation loop
            model.eval()
            correct = 0
            total = 0
            val_predicted_labels = []
            val_true_labels = []
    
            with torch.no_grad():
                for val_images, val_labels in val_loader:
                    val_images, val_labels = val_images.to(device), val_labels.unsqueeze(1).to(device)

                    val_outputs = model(val_images)

                    val_outputs = Sigmoid()(val_outputs)
                    predicted = (val_outputs.data >= decision).float()

                    total += val_labels.size(0)
                    correct += (predicted == val_labels).sum().item()
                    
                    val_predicted_labels.extend(predicted.cpu().numpy())
                    val_true_labels.extend(val_labels.cpu().numpy())
    
            val_accuracy = correct / total
            val_accuracies.append(val_accuracy)
            
            f1 = f1_score(val_true_labels, val_predicted_labels)
            f1_scores.append(f1)

        # Plotting the training and validation accuracy curves
        plt.plot(range(1, epoch + 2), train_accuracies, label='Training Accuracy')
        plt.plot(range(1, epoch + 2), val_accuracies, label='Validation Accuracy')
        plt.plot(range(1, epoch + 2), f1_scores, label='F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy / F1 score')
        plt.title(f"Training and Validation Accuracy Curves with F1-score of \n{name}")
        plt.legend()
        plt.savefig(f"/project_ghent/Master-Thesis/gif/images/trainandvalcurves_{name}.png")

        if epoch == num_epochs - 1:
            plt.show()
            
        # Clear current plot
        plt.clf()
        

def plot_confusionmatrix(model: Any, parameters: Dict[str, Any], name :str, mode: str="Validation", verbose: bool=False) -> None:
    """
    Plot the confusion matrix

    :param model: The model.
    :param parameters: The parameters for the training loop.
    :param name: The name of the model.
    :param mode: Indicating train, valid or test mode.
    :param verbose: True if showing extra information.
    """
    device = parameters["device"]
    train_loader = parameters["trainloader"]
    val_loader = parameters["valloader"]
    # test_loader = parameters["testloader"]
    catgs = parameters["categories"]
    decision = parameters["decision"]
    
    model.eval()
    
    all_labels = []
    all_predictions = []

    if mode == "Train":
        loader = train_loader
    elif mode == "Validation":
        loader = val_loader
    elif mode == "Test":
        loader = test_loader
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            val_outputs = model(images)

            val_outputs = Sigmoid()(val_outputs)
            
            if verbose:
                print(val_outputs, labels)
                
            predictions = (val_outputs >= decision).float()
    
            all_labels.extend(labels.to("cpu").numpy())
            all_predictions.extend(predictions.to("cpu").numpy())
    
    cm = confusion_matrix(all_labels, all_predictions)
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=catgs)
    disp.plot(cmap=plt.cm.Blues, values_format=".4g")
    
    plt.title(f"Confusion Matrix {mode} of \n{name}")
    plt.savefig(f"/project_ghent/Master-Thesis/gif/images/confusionmatrix_{name}.png")
    plt.show()


def plot_roc_curve(model, parameters: Dict[str, Any], name: str, mode: str = "Validation", verbose: bool = False) -> None:
    """
    Plot the ROC curve

    :param model: The model.
    :param parameters: The parameters for the training loop.
    :param name: The name of the model.
    :param mode: Indicating train, valid, or test mode.
    :param verbose: True if showing extra information.
    """
    device = parameters["device"]
    train_loader = parameters["trainloader"]
    val_loader = parameters["valloader"]
    # test_loader = parameters["testloader"]
    catgs = parameters["categories"]
    decision = parameters["decision"]
    
    model.eval()
    
    all_labels = []
    all_probs = []

    if mode == "Train":
        loader = train_loader
    elif mode == "Validation":
        loader = val_loader
    elif mode == "Test":
        loader = test_loader
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            val_outputs = model(images)
                
            val_probs = torch.sigmoid(val_outputs)
            
            if verbose:
                print(val_probs, labels)
                
            all_labels.extend(labels.to("cpu").numpy())
            all_probs.extend(val_probs.to("cpu").numpy())
    
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve {mode} of \n{name}')
    plt.legend(loc="lower right")
    plt.savefig(f"/project_ghent/Master-Thesis/gif/images/roc_curve_{name}.png")
    plt.show()


def calculate_precision_recall(model: Any, parameters: Dict[str, Any], name: str, decision: int) -> Tuple[int, int, int]:
    """
    Calculate the precision, recall and F1 score.
    
    :param model: The model.
    :param parameters: The parameters for the training loop.
    :param name: The name of the model.
    :param decision: The decision threshold.
    :return precision: The precision.
    :return recall: The recall.
    :return F1: The F1 score.
    """
    device = parameters["device"]
    val_loader = parameters["valloader"]
    
    model.eval()
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)

            outputs = Sigmoid()(outputs)
            predicted = (outputs >= decision).float()
            
            true_positives += ((predicted == 1) & (labels == 1)).sum().item()
            false_positives += ((predicted == 1) & (labels == 0)).sum().item()
            false_negatives += ((predicted == 0) & (labels == 1)).sum().item()
    
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives != 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives != 0 else 0
    F1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    
    return precision, recall, F1

def show_measures_for_decision_boundaries(model: Any, parameters: Dict[str, Any], name: str) -> None:
    """
    Print the measures for all decision boundaries in steps of 0.1.
    
    :param model: The model.
    :param parameters: The parameters for the training loop.
    :param name: The name of the model.
    """
    decs = [i / 10 for i in range(11)]
    for dec in decs:
        precision, recall, F1 = calculate_precision_recall(model, parameters, name, dec)
        print(f"{dec} -> Precision: {precision}, Recall: {recall}, F1: {F1}")
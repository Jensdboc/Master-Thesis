import os
import os.path
import yaml
import cv2
import torch
import mediapipe as mp

from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))

yaml_file_path = os.path.join(current_dir, "data.yaml")

with open(yaml_file_path, "r") as file:
    data = yaml.safe_load(file)
    classes = data["names"]
    num_classes = len(classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} as device")

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


def extract_face_landmarks_path(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
        else:
            return None
    return landmarks


def extract_face_landmark_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as face_mesh:
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
        else:
            return None
    return landmarks


def train_and_plot(model, parameters):
    model = model.to(device)

    # Set the number of epochs
    # num_epochs = 30
    num_epochs = parameters["epochs"]
    train_loader = parameters["trainloader"]
    val_loader = parameters["valloader"]
    criterion = parameters["criterion"]
    optimizer = parameters["optimizer"]

    # Lists to store training and validation accuracies
    train_accuracies = []
    val_accuracies = []

    # Training loop with tqdm progress bar
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode

        # Initialize the progress bar
        with tqdm(
            total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"
        ) as pbar:
            correct_train = 0
            total_train = 0

            for csv, labels in train_loader:
                csv, labels = csv.to(device), labels.to(device)
                # Forward pass
                outputs = model(csv)
                loss = criterion(outputs, labels)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix({"Loss": loss.item()})

                # Calculate training accuracy
                _, predicted_train = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted_train == labels).sum().item()

            # Calculate and store training accuracy
            train_accuracy = correct_train / total_train
            train_accuracies.append(train_accuracy)

            # Validation loop
            model.eval()  # Set the model to evaluation mode
            correct = 0
            total = 0

            with torch.no_grad():
                for val_images, val_labels in val_loader:
                    val_images, val_labels = val_images.to(device), val_labels.to(
                        device
                    )
                    val_outputs = model(val_images)
                    _, predicted = torch.max(val_outputs.data, 1)
                    total += val_labels.size(0)
                    correct += (predicted == val_labels).sum().item()

            # Calculate validation accuracy
            val_accuracy = correct / total
            val_accuracies.append(val_accuracy)

    # Plotting the training and validation accuracy curves
    plt.plot(range(1, num_epochs + 1), train_accuracies, label="Training Accuracy")
    plt.plot(range(1, num_epochs + 1), val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy Curves")
    plt.legend()
    plt.show()
    return train_accuracies, val_accuracies


def plot_confusionmatrix(model, test_loader):
    # Assuming you have a DataLoader named validation_loader and a trained model
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            all_labels.extend(labels.to("cpu").numpy())
            all_predictions.extend(predictions.to("cpu").numpy())

    cm = confusion_matrix(all_labels, all_predictions)
    class_names = [
        str(i) for i in data["names"]
    ]

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format=".4g")

    plt.title("Confusion Matrix Test")
    plt.show()


blendnames = [
    "_neutral",
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "eyeLookDownLeft",
    "eyeLookDownRight",
    "eyeLookInLeft",
    "eyeLookInRight",
    "eyeLookOutLeft",
    "eyeLookOutRight",
    "eyeLookUpLeft",
    "eyeLookUpRight",
    "eyeSquintLeft",
    "eyeSquintRight",
    "eyeWideLeft",
    "eyeWideRight",
    "jawForward",
    "jawLeft",
    "jawOpen",
    "jawRight",
    "mouthClose",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    "noseSneerLeft",
    "noseSneerRight",
]

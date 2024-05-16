import utils
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

seed = 42

print("Parsing arguments...")

parser = argparse.ArgumentParser()

parser.add_argument("-b", "--batch", type=int, default=16)
parser.add_argument("-s", "--skip", type=int, default=1)
parser.add_argument("-e", "--epoch", type=int, default=5)
parser.add_argument("-lr", "--learningrate", type=float, default=1e-4)
parser.add_argument("-pos", "--posweight", type=float, default=0.378)
parser.add_argument("-w", "--workers", type=int, default=8)
parser.add_argument("-is", "--inputsize", type=int, default=52)
parser.add_argument("-hs", "--hiddensize", type=int, default=512)
parser.add_argument("-nl", "--numlayers", type=int, default=1)
parser.add_argument("-name", "--name", type=str, default="")

args = parser.parse_args()

batch_size = args.batch
skipframes = args.skip
num_epochs = args.epoch
lr = args.learningrate
num_workers = args.workers
pos_weight = args.posweight
INPUT_SIZE = args.inputsize
HIDDEN_SIZE = args.hiddensize
NUM_LAYERS = args.numlayers

AMOUNT_OF_FRAMES = 120
name = f"blendshape_b{batch_size}s{skipframes}e{num_epochs}lr{lr}is{INPUT_SIZE}hs{HIDDEN_SIZE}nl{NUM_LAYERS}_{args.name}"

print("Retrieving datasets and loaders...")

train_ds, val_ds, test_ds = utils.get_dataset(skipframes=args.skip, name=args.name)

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, worker_init_fn=np.random.seed(seed), num_workers=num_workers)
val_dl = DataLoader(val_ds, batch_size=2*batch_size, shuffle=False, num_workers=num_workers)  
test_dl = DataLoader(test_ds, batch_size=2*batch_size, shuffle=False, num_workers=num_workers)

print("Initializing parameters...")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = utils.RecipeBlendshapeModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)

criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor([pos_weight]).to(device))
optimizer = optim.Adam(model.parameters(), lr=lr)

parameters = {"device": device, 
              "epochs": num_epochs,
              "trainloader": train_dl,
              "valloader": val_dl,
              "testloader": test_dl,
              "criterion": criterion,
              "optimizer": optimizer,
              "batch_size": batch_size,
              "amount_of_frames": AMOUNT_OF_FRAMES // skipframes,
              "categories":  utils.categories.keys(),
              "decision": 0.5}

print("Training...")

utils.train_and_plot(model, parameters, name)

print("Plotting confusion matrix...")

utils.plot_confusionmatrix(model, parameters, name, mode="Validation", verbose=False)

print("Plotting ROC curve...")

utils.plot_roc_curve(model, parameters, name, mode="Validation", verbose=False)

print("Saving model...")

torch.save(model, f"/project_ghent/Master-Thesis/ownModelNotebooks/models/blendshape_{name}.pth")

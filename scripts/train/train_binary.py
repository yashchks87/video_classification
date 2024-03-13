import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import wandb
import sys
sys.path.append('../scripts/')  
from train.save_model import save_model
from train.metrics_binary import precision_binary, recall_binary

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_binary_fn(model, train_loader, val_loader, save_weights_path, load_model = None, wandb_dict = None, epochs = 10):
    if wandb_dict == None:
        wandb.init(project='video_classification')
    else:
        wandb.init(project='video_classification', config=wandb_dict)
    data_loaders = {
        'train': train_loader,
        'val': val_loader
    }
    # if load_model != None:
    #     model = model.load_state_dict(torch.load(load_model), strict=False)
    if load_model != None:
        model.load_state_dict(torch.load(load_model)['model_state_dict'], strict=False)
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(epochs): 
        train_loss, val_loss = 0.0, 0.0
        train_prec, val_prec = 0.0, 0.0
        train_rec, val_rec = 0.0, 0.0
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss, running_prec, running_rec = 0.0, 0.0, 0.0
            with tqdm(data_loaders[phase], unit="batch") as tepoch:
                for frames, label in tepoch:
                    frames, label = frames.to(device), label.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(frames)
                        loss = F.binary_cross_entropy(F.sigmoid(outputs), label)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item()
                    running_prec += precision_binary(outputs, label).item()
                    running_rec += recall_binary(outputs, label).item()
                    tepoch.set_postfix(loss=running_loss, prec = running_prec, rec = running_rec)
            if phase == 'train':
                train_loss = running_loss / len(train_loader)
                train_prec = running_prec / len(train_loader)
                train_rec = running_rec / len(train_loader)
            else:
                val_loss = running_loss / len(val_loader)
                val_prec = running_prec / len(val_loader)
                val_rec = running_rec / len(val_loader)
        wandb.log({
                    'train_loss': train_loss, 
                    'val_loss': val_loss,
                    'train_prec': train_prec,
                    'val_prec': val_prec,
                    'train_rec': train_rec,
                    'val_rec': val_rec
                })
        print(f'Epoch {epoch}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss}')
        print(f'Epoch {epoch}/{epochs} | Train Prec: {train_prec} | Val Prec: {val_prec}')
        print(f'Epoch {epoch}/{epochs} | Train Rec: {train_rec} | Val Rec: {val_rec}')
        save_model(model, epoch, optimizer, False, save_weights_path)
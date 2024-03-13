import torch
import os

def save_model(model, epoch, optimizer, multiple_gpus, save_path):
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    torch.save({
        'epoch': epoch,
        'model_state_dict':  model.module.state_dict() if multiple_gpus == True else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, f'{save_path}/model_{epoch}.pth')
    print(f'Weight saved for epoch: {epoch}')

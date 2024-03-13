import torch
import torch.nn.functional as F

def precision_binary(outputs, labels, threshold = 0.5):
    # recall = TP / (TP + FN)
    outputs = F.sigmoid(outputs)
    outputs_adjusted = torch.where(outputs > threshold, 1.0, 0.0)
    tp = torch.logical_and(outputs_adjusted == 1.0, labels == 1.0).sum()
    fn = torch.logical_and(outputs_adjusted == 0.0, labels == 1.0).sum()
    return tp / (tp + fn + 1e-7)

def recall_binary(outputs, labels, threshold = 0.5):
    # recall = TP / (TP + FP)
    outputs = F.sigmoid(outputs)
    outputs_adjusted = torch.where(outputs > threshold, 1.0, 0.0)
    tp = torch.logical_and(outputs_adjusted == 1.0, labels == 1.0).sum()
    fp = torch.logical_and(outputs_adjusted == 1.0, labels == 0.0).sum()
    return tp / (tp + fp + 1e-7)

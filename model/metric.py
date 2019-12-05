import torch
import numpy as np 
from sklearn.metrics import roc_auc_score, average_precision_score

def aupr(output, target):
    output = average_precision_score(target.clone().detach().cpu().numpy(), output.clone().detach().cpu().numpy())
    return output

def auroc(output, target):
    output = roc_auc_score(target.clone().detach().cpu().numpy(), output.clone().detach().cpu().numpy())    
    return output

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        target_class = torch.argmax(target, dim = 1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target_class).item()
    return correct / len(target)



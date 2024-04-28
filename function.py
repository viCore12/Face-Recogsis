import torch
import torch.nn as nn
from torcheval.metrics.functional import multiclass_f1_score

def loss_fn(output, target, num_labels):
    lfn = nn.CrossEntropyLoss()
    loss = lfn(output, target.long())
    return loss

def acc_fn(output, target, num_labels):
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).sum().item()
    accuracy = correct / target.size(0)
    return accuracy

def f1_fn(output, target, num_labels):
    f1 = multiclass_f1_score(output, target.long(), num_classes=num_labels, average="macro")
    return f1
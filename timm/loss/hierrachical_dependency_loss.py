import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalDependencyLoss(nn.Module):
    def __init__(self, La, parents, device):
        super(HierarchicalDependencyLoss, self).__init__()
        self.La = La
        self.parents = torch.tensor(parents).to(device)

    def forward(self, y_pred, y_true, layer):
        y_pred_child_layer_img = torch.argmax(torch.mul(y_pred, self.La[layer,:]))
        y_pred_parent_layer_img = torch.argmax(torch.mul(y_pred, self.La[layer - 1,:]))
        Dl = torch.tensor(0, dtype=torch.float) if torch.argmax(self.parents[y_pred_child_layer_img,:]) == y_pred_parent_layer_img else torch.tensor(1, dtype=torch.float)
        return torch.exp(Dl) - 1
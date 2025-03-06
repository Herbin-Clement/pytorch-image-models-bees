import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.loss.hierrachical_dependency_loss import HierarchicalDependencyLoss

class TaxonomicLoss(nn.Module):
    def __init__(self, La, layers, parents, alpha_layer, device):
        super(TaxonomicLoss, self).__init__()
        self.h = len(layers)
        self.La = torch.tensor(La).to(device)
        self.layers = [torch.tensor(layers[i]).to(device) for i in range(self.h)]
        self.alpha_layer = torch.tensor(alpha_layer).to(device)
        self.cce = nn.CrossEntropyLoss()
        self.hd = HierarchicalDependencyLoss(self.La, parents, device)


    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        batch_size = y_pred.shape[0]
        loss = 0
        for i in range(batch_size):
            loss_img = 0
            for layer in range(1, self.h):
                y_pred_layer_img = torch.softmax(y_pred[i, self.layers[layer]], dim=0)
                y_true_layer_img = y_true[i, self.layers[layer]]
                cce_loss = self.cce(y_pred_layer_img, y_true_layer_img)
                d_loss = self.hd(y_pred[i, :], y_true[i, :], layer)
                loss_img += (cce_loss + d_loss) * self.alpha_layer[layer]
                
            loss += loss_img

        return loss / batch_size
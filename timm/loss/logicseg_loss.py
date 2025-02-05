import torch
import torch.nn as nn
import torch.nn.functional as F

from logicseg.c_rule_loss import CRuleLoss
from logicseg.d_rule_loss import DRuleLoss
from logicseg.e_rule_loss import ERuleLoss
from binary_cross_entropy import BinaryCrossEntropy

class LogicSegLoss(nn.Module):
    def __init__(self, H_raw, P_raw, M_raw, alpha_c, alpha_d, alpha_e, alpha_bce): # H_raw is a np array
        super(LogicSegLoss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.c_rule = CRuleLoss(H_raw)
        self.alpha_c = alpha_c

        self.d_rule = DRuleLoss(H_raw)
        self.alpha_d = alpha_d

        self.e_rule = ERuleLoss(P_raw, M_raw)
        self.alpha_e = alpha_e

        # self.bce = BinaryCrossEntropy()
        self.alpha_bce = alpha_bce
  
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        with_print = False
        if with_print:
            print("L_c =", self.c_rule(y_pred, y_true).item())
            print("L_d =", self.d_rule(y_pred, y_true).item())
            print("L_e =", self.e_rule(y_pred, y_true).item())
            # print(self.bce(y_pred, y_true))
            print("L_BCE =", F.binary_cross_entropy(y_pred, y_true).item())

        batch_size = y_pred.shape[0]
        batch_losses = self.alpha_c * self.c_rule(y_pred, y_true) + \
                       self.alpha_d * self.d_rule(y_pred, y_true) + \
                       self.alpha_e * self.e_rule(y_pred, y_true) + \
                       self.alpha_bce * (F.binary_cross_entropy(y_pred, y_true) / batch_size)
        return batch_losses
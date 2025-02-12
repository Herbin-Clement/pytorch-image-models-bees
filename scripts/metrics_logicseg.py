from scripts.logic_seg_utils import *

""" Métriques de logicSeg
"""


def accuracy_logicseg(output, target, label_matrix, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # print("output shape", output.shape)
    # print("target shape", target.shape)
    # print("topk", topk)
    probas_branches_input = get_predicted_branches(output, label_matrix) # taille (nb_pred, nb_feuilles)
    probas_branches_target = get_predicted_branches(target, label_matrix)
    # top 1
    _, indices_branches_in = probas_branches_input.topk(1, dim=1) # (nb_pred, top_k), (nb_pred, top_k)
    _, indices_branches_target = probas_branches_target.topk(1, dim=1) # (nb_pred, top_k), (nb_pred, top_k)
    acc1 = (indices_branches_in == indices_branches_target).sum().item() / indices_branches_in.size(0)
    # top 5
    _, indices_branches_in = probas_branches_input.topk(5, dim=1) # (nb_pred, top_k), (nb_pred, top_k)
    _, indices_branches_target = probas_branches_target.topk(5, dim=1) # (nb_pred, top_k), (nb_pred, top_k)
    acc5 = (indices_branches_in == indices_branches_target).sum().item() / indices_branches_in.size(0)
    return acc1, acc5 

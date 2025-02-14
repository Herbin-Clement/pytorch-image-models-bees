import os

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np

from scripts.hierarchy_better_mistakes_utils import *

def show_results_from_csv_summary(filename):
    """
    Show results from summary csv produced by TIMM.
    """
    matplotlib.use('TkAgg')

    data = pd.read_csv(filename)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(data['epoch'], data['train_loss'], label='Train Loss', color='blue', linestyle='--')
    ax1.plot(data['epoch'], data['eval_loss'], label='Eval Loss', color='red')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.set_title("Training and Evaluation Loss")
    ax1.grid()

    fig, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(data['epoch'], data['eval_top1'], label='Eval Top-1 Accuracy', color='green')
    ax2.plot(data['epoch'], data['eval_top5'], label='Eval Top-5 Accuracy', color='purple')
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend(loc="lower right")
    ax2.set_title("Evaluation Accuracy")
    ax2.grid()

    plt.show()
    plt.imsave()

def show_results_from_csv_summary_cce_hce(filename_cce, filename_hce, folder="output/img"):
    """
    Show results from summary csv produced by TIMM with HCE and CCE.
    """
    matplotlib.use('TkAgg')

    data_cce = pd.read_csv(filename_cce)
    data_hce = pd.read_csv(filename_hce)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(data_cce['epoch'], data_cce['train_loss'], label='CCE Train Loss', color='red', linestyle='--')
    ax1.plot(data_cce['epoch'], data_cce['eval_loss'], label='CCE Eval Loss', color='orange')
    ax1.plot(data_hce['epoch'], data_hce['train_loss'], label='HCE Train Loss', color='green', linestyle='--')
    ax1.plot(data_hce['epoch'], data_hce['eval_loss'], label='HCE Eval Loss', color='lime')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.set_title("Training and Evaluation Loss")
    ax1.grid()
    fig.savefig(os.path.join(folder, f"loss_summary_cce_hce"))

    fig, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(data_cce['epoch'], data_cce['eval_top1'], label='CCE Eval Top-1 Accuracy', color='red')
    ax2.plot(data_cce['epoch'], data_cce['eval_top5'], label='CCE Eval Top-5 Accuracy', color='orange')
    ax2.plot(data_hce['epoch'], data_hce['eval_top1'], label='HCE Eval Top-1 Accuracy', color='green')
    ax2.plot(data_hce['epoch'], data_hce['eval_top5'], label='HCE Eval Top-5 Accuracy', color='lime')
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend(loc="lower right")
    ax2.set_title("Evaluation Accuracy")
    ax2.grid()
    fig.savefig(os.path.join(folder, f"acc_summary_cce_hce"))

    plt.show()

def show_results_from_csv_summary_cce_hce_alpha(filename_cce, filename_hce_0_1, filename_hce_0_5, folder="output/img"):
    """
    Show results from summary csv produced by TIMM with HCE and CCE.
    """
    matplotlib.use('TkAgg')

    data_cce = pd.read_csv(filename_cce)
    data_hce_0_1 = pd.read_csv(filename_hce_0_1)
    data_hce_0_5 = pd.read_csv(filename_hce_0_5)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(data_cce['epoch'], data_cce['train_loss'], label='CCE Train Loss', color='red', linestyle='--')
    ax1.plot(data_cce['epoch'], data_cce['eval_loss'], label='CCE Eval Loss', color='orange')
    ax1.plot(data_hce_0_1['epoch'], data_hce_0_1['train_loss'], label='HCE (alpha=0.1) Train Loss', color='green', linestyle='--')
    ax1.plot(data_hce_0_1['epoch'], data_hce_0_1['eval_loss'], label='HCE (alpha=0.1) Eval Loss', color='lime')
    ax1.plot(data_hce_0_5['epoch'], data_hce_0_5['train_loss'], label='HCE (alpha=0.5) Train Loss', color='blue', linestyle='--')
    ax1.plot(data_hce_0_5['epoch'], data_hce_0_5['eval_loss'], label='HCE (alpha=0.5) Eval Loss', color='skyblue')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.set_title("Training and Evaluation Loss")
    ax1.grid()
    fig.savefig(os.path.join(folder, f"loss_summary_cce_hce"))

    fig, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(data_cce['epoch'], data_cce['eval_top1'], label='CCE Eval Top-1 Accuracy', color='red')
    ax2.plot(data_cce['epoch'], data_cce['eval_top5'], label='CCE Eval Top-5 Accuracy', color='orange')
    ax2.plot(data_hce_0_1['epoch'], data_hce_0_1['eval_top1'], label='HCE (alpha=0.1) Eval Top-1 Accuracy', color='green')
    ax2.plot(data_hce_0_1['epoch'], data_hce_0_1['eval_top5'], label='HCE (alpha=0.1) Eval Top-5 Accuracy', color='lime')
    ax2.plot(data_hce_0_5['epoch'], data_hce_0_5['eval_top1'], label='HCE (alpha=0.5) Eval Top-1 Accuracy', color='blue')
    ax2.plot(data_hce_0_5['epoch'], data_hce_0_5['eval_top5'], label='HCE (alpha=0.5) Eval Top-5 Accuracy', color='skyblue')
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend(loc="lower right")
    ax2.set_title("Evaluation Accuracy")
    ax2.grid()
    fig.savefig(os.path.join(folder, f"acc_summary_cce_hce"))

    plt.show()

def load_confusion_matrix(filename):
    """
    Load confusion matrix from a txt file.
    """
    cm = np.loadtxt(filename)
    cm = cm.astype(int)
    return cm

def save_confusion_matrix(cm, output_filename, classes, folder="output/img"):
    """
    Save confusion matrix image.
    """
    figsize = len(classes) / 2 if len(classes) > 10 else 5
    plt.figure(figsize=(figsize, figsize))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Prédictions")
    plt.ylabel("Vérités")
    plt.title("Matrice de Confusion")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, output_filename))
    
    return cm

def load_classnames(filename):
    """
    Load classnames from a file.
    """
    classes = []
    with open(filename, 'r') as f:
        data = f.readlines()
        for line in data:
            classes.append(line.replace('\n', ''))
    return classes

def calculate_metrics(cm):
    """
    Compute F1-score, Precision and Recall from a confusion matrix.
    """
    TP = np.diag(cm)    
    FP = np.sum(cm, axis=0) - TP    
    FN = np.sum(cm, axis=1) - TP    
    precision = np.divide(TP, TP + FP, where=(TP + FP) != 0)
    recall = np.divide(TP, TP + FN, where=(TP + FN) != 0)    
    f1_score = np.divide(2 * precision * recall, precision + recall, where=(precision + recall) != 0)
    tot_pred = np.sum(cm, axis=0)    
    tot_true = np.sum(cm, axis=1)
    return precision, recall, f1_score, tot_pred, tot_true, TP, FP, FN

def save_metrics(cm, folder, filename, classes, hierarchy_name):
    """
    Save metrics from a confusion matrix in a file.
    """
    precision, recall, f1_score, tot_pred, tot_true, TP, FP, FN = calculate_metrics(cm)

    # Création d'un DataFrame Pandas
    df = pd.DataFrame({
        "Classe": classes, 
        "Etage": [hierarchy_name for i in range(len(classes))],
        "Pred": tot_pred,
        "True": tot_true,
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "Précision": precision, 
        "Rappel": recall,
        "F1-score": f1_score
    })

    # Ajout des moyennes globales
    df.loc["Moyenne Macro"] = ["Moyenne", hierarchy_name, np.mean(tot_pred), np.mean(tot_true), np.mean(TP), np.mean(FP), np.mean(FN), np.mean(precision), np.mean(recall), np.mean(f1_score)]

    # Sauvegarde en CSV
    df.to_csv(os.path.join(folder, filename), index=False)
    return df

def get_id_from_nodes(hierarchy_lines):
    """
    Get Nodes and Leafs ID.
    """
    h = len(hierarchy_lines[0])
    nodes = []
    for i in range(h-1):
        for line in hierarchy_lines:
            if line[i] not in nodes:
                nodes.append(line[i])
    leafs = []
    for line in hierarchy_lines:
        leafs.append(line[h-1])
    nodes_to_id = {node: i for i, node in enumerate(nodes)}
    leafs_to_id = {leaf: i for i, leaf in enumerate(leafs)}
    return nodes, leafs, nodes_to_id, leafs_to_id

def get_parent_confusion_matrix(cm, classes, parents):
    next_classes = set()
    for class_ in classes:
        next_classes.add(parents[class_])
    next_classes = list(next_classes)
    next_classes_id = {class_: i for i, class_ in enumerate(next_classes)}
    
    next_cm = np.zeros((len(next_classes),len(next_classes)), dtype=int)
    for i, class_1 in enumerate(classes):
        for j, class_2 in enumerate(classes):
            next_class_1 = parents[class_1]
            next_class_2 = parents[class_2]
            next_class_1_id = next_classes_id[next_class_1]
            next_class_2_id = next_classes_id[next_class_2]
            next_cm[next_class_1_id, next_class_2_id] += cm[i,j]

    return next_cm, next_classes


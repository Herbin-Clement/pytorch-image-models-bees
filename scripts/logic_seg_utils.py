import numpy as np
import pickle
import pandas as pd
import torch

def get_layer_matrix(path_to_csv_tree, verbose=False):
  csv = pd.read_csv(path_to_csv_tree)
  unique_nodes = pd.unique(csv.values.ravel())
  unique_nodes = unique_nodes[~pd.isnull(unique_nodes)]  # On enlève les NaN au cas ou
  node_to_index = {node: idx for idx, node in enumerate(unique_nodes)}
  # if verbose:
  #   print(node_to_index)
  La = np.zeros((len(csv.columns), len(unique_nodes)))
  for _, row in csv.iterrows():
     for layer, node in enumerate(row):
        La[layer, node_to_index[node]] = 1

  if verbose:
     print(La)

  return La

def get_tree_matrices(path_to_csv_tree, verbose=False):
  csv = pd.read_csv(path_to_csv_tree)
  unique_nodes = pd.unique(csv.values.ravel())
  unique_nodes = unique_nodes[~pd.isnull(unique_nodes)]  # On enlève les NaN au cas ou
  node_to_index = {node: idx for idx, node in enumerate(unique_nodes)}

  n = len(unique_nodes)
  H = np.zeros((n, n), dtype=int)

  # On parcours le csv lignes par lignes (donc chemin par chemin dans l'arbre)
  for _, row in csv.iterrows():
      for parent, child in zip(row[:-1], row[1:]):
          if pd.notna(parent) and pd.notna(child):
              H[node_to_index[parent], node_to_index[child]] = 1
  
  if verbose:
    print("Nodes:", unique_nodes)
    print("H:\n", H)

  peer_matrix = np.matmul(H.T,H)
  peer_matrix = np.maximum(peer_matrix - np.eye(n),0)

  M = np.sum(peer_matrix, axis=0) # peer_matrix est symétrique donc l'axe n'est pas important
  
  return H, peer_matrix, M

def create_class_to_labels(path_to_csv_tree, path_to_temporary_class_to_labels_file, verbose=False):
  label_matrix, _, index_to_node = get_label_matrix(path_to_csv_tree, verbose)

  class_to_labels = get_class_to_label(label_matrix, index_to_node, verbose)

  # Sauvegarder dans un fichier .pkl
  with open(path_to_temporary_class_to_labels_file, "wb") as f:
      pickle.dump(class_to_labels, f)  # Écriture binaire

def get_label_matrix(path_to_csv_tree, verbose=False):
  # print(path_to_csv_tree)
  csv = pd.read_csv(path_to_csv_tree)
  unique_nodes = pd.unique(csv.values.ravel())
  unique_nodes = unique_nodes[~pd.isnull(unique_nodes)]  # On enlève les NaN au cas ou
  node_to_index = {node: idx for idx, node in enumerate(unique_nodes)}
  index_to_node = {v: k for k, v in node_to_index.items()}
  n = len(unique_nodes)
  label_matrix = np.zeros((len(csv),n), dtype=int)
  # On parcours le csv lignes par lignes (donc chemin par chemin dans l'arbre)
  for i, row in csv.iterrows():
      for node in row:
        label_matrix[i,node_to_index[node]] = 1
  
  if(verbose):
    print('labels_matrix')
    print(label_matrix)
  return label_matrix, node_to_index, index_to_node

def get_class_to_label(label_matrix, index_to_node, verbose=False):
  class_to_labels={}
  # On parcours le csv lignes par lignes (donc chemin par chemin dans l'arbre)
  for i in range(label_matrix.shape[0]):
      current_branch = label_matrix[i]
      class_index = np.where(current_branch == 1)[0][-1]
      class_name = index_to_node[class_index]
      class_to_labels[class_name] = current_branch.tolist()
  if(verbose):
    print('class_to_labels')
    print(class_to_labels)
    print("class_to_labels 1", class_to_labels)
  return class_to_labels

# pred: sigmoid(y_pred)
# on veut: tenseur de taille (nb_branches,) contenant les probas de toutes les branches
# sortie: shape = (nb_pred, nb_branches)
def get_logicseg_predictions(pred, label_matrix, device):
  label_matrix = torch.tensor(label_matrix, dtype=torch.float32).to(device) #torch tensor
  nb_pred = pred.shape[0]
  nb_feuilles = label_matrix.shape[0]
  probas_branches = torch.empty(size=(nb_pred, nb_feuilles), dtype=torch.float32)
  for i in range(nb_pred):
    pred_rep = pred[i,:].repeat(nb_feuilles, 1)
    probas_branches[i,:] = torch.sum(pred_rep*label_matrix, dim=1)
  return probas_branches

# most_probables_branches: indices des k branches les plus probables dans l'ordre décroissant de probabilité
# return: les k labels textuels associés
# most (nb_pred, top_k)
def get_branches_label(most_probable_branches_indices_in, most_probable_branches_indices_target, class_to_label):
  predicted_classes = np.empty(most_probable_branches_indices_in.shape, dtype=object)
  classes = list(class_to_label.keys())

  for p in range(most_probable_branches_indices_in.shape[0]):
    for i in range(most_probable_branches_indices_in.shape[1]):
      predicted_classes[p,i] = "pred: " + classes[most_probable_branches_indices_in[p,i]] + ", label:  " + classes[most_probable_branches_indices_target[p,i]]
  return predicted_classes

if __name__ == "__main__":
   path_to_csv_tree = "data/simple/hierarchy_test.csv"
   get_tree_matrices(path_to_csv_tree, verbose=True)

def get_csv_header(file_path):
    import csv
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Read the first row
    return header
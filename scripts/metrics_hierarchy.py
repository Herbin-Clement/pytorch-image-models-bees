import torch
from logic_seg_utils import get_logicseg_predictions

class MetricsLabels:
    """Classe pour stocker les labels des différentes métriques"""

    accuracy_top1 = "accuracy top 1"
    accuracy_top5 = "accuracy top 5"
    hierarchical_distance_mistakes = "hierarchical distance mistakes"
    topk_hierarchical_distance_mistakes = "hierarchical distance mistakes"
    c_rule_respect = "respect of the c rule"
    d_rule_respect = "respect of the d rule"
    e_rule_respect = "respect of the e rule"


class MetricsHierarchy:
    """Classe pour calculer et stocker différentes métriques de performance d'une IA."""

    def __init__(self, H : torch.Tensor):
        """Initialise le dictionnaire pour stocker les métriques."""
        self.metrics = {
            MetricsLabels.accuracy_top1: None,
            MetricsLabels.accuracy_top5: None,
            MetricsLabels.hierarchical_distance_mistakes: None,
            MetricsLabels.topk_hierarchical_distance_mistakes: None,
            MetricsLabels.c_rule_respect: None,
            MetricsLabels.d_rule_respect: None,
            MetricsLabels.e_rule_respect: None,
        }

        self.H = H
    
    
    def lca_height(self, node1 : int, node2: int):
        """Trouve la distance qui sépare les nœuds du Lowest Common Ancestor.

        :param node1: Premier nœud.
        :param node2: Deuxième nœud.
        :return: (LCA, distance de node1 au LCA, distance de node2 au LCA)"""
           
        distance = 0
        current_nodes = [-1,-2]
        
        while current_nodes[0] != current_nodes[1]:
            parents = torch.where(self.H[:, list(current_nodes)] == 1)[0].tolist()  # Trouver les parents

            if (len(parents) == 0):
                raise Exception("The node have no parent")
            if (len(parents) > 2):
                raise Exception("2 or more parents for one node")
            
            current_nodes = parents  # Continuer avec les nouveaux parents
            if (current_nodes[0] != current_nodes[1]):
                distance += 1

        return distance
    

    def hierarchical_distance_mistake(self, output, target, label_matrix):
        """
        Calcule la distance hierarchique des erreurs

        Args:
            output (torch.Tensor): Prédictions du modèle (logits).
            target (torch.Tensor): Labels réels.
            label_matrix (torch.Tensor): Matrice des labels.
        """
        probas_branches_input = get_logicseg_predictions(output, label_matrix)
        probas_branches_target = get_logicseg_predictions(target, label_matrix)

        _, indices_branches_in = probas_branches_input.topk(1, dim=1)
        _, indices_branches_target = probas_branches_target.topk(1, dim=1)

        # Initialiser la distance totale
        total_distance = 0.0

        for i in range(target.size(0)):  # Boucle sur tous les exemples
            pred_class = indices_branches_in[i].item()
            true_class = indices_branches_target[i].item()

            if pred_class != true_class:
                distance = self.lca_height(pred_class, true_class)
                total_distance += distance

        self.metrics[MetricsLabels.hierarchical_distance_mistakes] = total_distance / target.size(0)

    def topk_hierarchical_distance_mistake(self, output, target, label_matrix, k=5):
        """
        Calcule la distance hiérarchique moyenne des erreurs pour les k meilleures prédictions.

        Args:
            output (torch.Tensor): Prédictions du modèle (logits), de taille (batch_size, num_classes).
            target (torch.Tensor): Labels réels, de taille (batch_size,).
            label_matrix (torch.Tensor): Matrice des labels.
            k (int): Nombre de classes à considérer dans le top-k.

        Returns:
            float: Distance hiérarchique moyenne des erreurs pour le top-k.
        """
        # Obtenir les probabilités des branches pour l'input et la cible
        probas_branches_input = get_logicseg_predictions(output, label_matrix)
        probas_branches_target = get_logicseg_predictions(target, label_matrix)

        # Obtenir les indices des k meilleures prédictions et de la cible
        _, indices_branches_in = probas_branches_input.topk(k, dim=1)  # (batch_size, k)
        _, indices_branches_target = probas_branches_target.topk(1, dim=1)  # (batch_size, 1)
        
        # Initialiser la distance totale
        total_distance = 0.0

        for i in range(target.size(0)):  # Boucle sur tous les exemples
            true_class = indices_branches_target[i].item()  # Vraie classe
            distances = []

            for j in range(k):  # Comparer aux k classes les plus probables 
                pred_class = indices_branches_in[i, j].item()
                if pred_class != true_class: 
                    distance = self.lca_height(pred_class, true_class)
                    distances.append(distance)

            # Prendre la moyenne des distances des top-k prédictions
            total_distance += sum(distances) / k

        # Stocker le résultat
        self.metrics[MetricsLabels.topk_hierarchical_distance_mistakes.format(k)] = total_distance / target.size(0)

    def c_rule_respect_percentage(self, output: torch.Tensor, target, label_matrix: torch.Tensor):
        """
        Calcule le pourcentage d'échantillons respectant la C-Rule.

        Args:
            output (torch.Tensor): Matrice des prédictions du modèle (batch_size, num_classes).
            label_matrix (torch.Tensor): Matrice one-hot des labels réels (batch_size, num_classes).

        Returns:
            float: Pourcentage des échantillons respectant la C-Rule.
        """
        batch_size, num_classes = output.shape

        # Seuil pour binariser les prédictions (0 ou 1)
        output_pred = (output > 0.5).float()  # Matrice binaire (batch_size, num_classes)


        # Calcul des activations des super-classes via la matrice H (Hiérarchie)
        H = self.H.float()  # Matrice hiérarchique (num_classes, num_classes)
        Hs = ((output_pred.T).repeat(1,num_classes) == 1) & (H == 1) 
        Hs = torch.sum(Hs.float(),dim=0)

        enfants  = torch.sum(H,dim=0)
        enfants_batch = enfants.repeat(batch_size,1)  # (batch_size, num_classes)

        # Vérifier que si une classe est activée, sa super-classe l'est aussi
        # Si une classe est activée mais pas sa super-classe, cela viole la règle
        violation_mask = (output_pred > 0) & (Hs == 0) & (enfants_batch != 0)  # (num_classes, batch_size)
         
        # Compter les échantillons respectant la règle (aucune violation)
        batch_respect = (torch.sum(violation_mask, dim=1) == 0).float()  # (batch_size,)

        # Calcul du pourcentage d'échantillons respectant la C-Rule
        total_respect = torch.mean(batch_respect)
        self.metrics[MetricsLabels.c_rule_respect] = total_respect



    def d_rule_respect_percentage(self, output: torch.Tensor, target, label_matrix: torch.Tensor):
        """
        Calcule le pourcentage d'échantillons respectant la D-Rule.

        Args:
            output (torch.Tensor): Matrice des prédictions du modèle (batch_size, num_classes).
            label_matrix (torch.Tensor): Matrice one-hot des labels réels (batch_size, num_classes).

        Returns:
            float: Pourcentage des échantillons respectant la D-Rule.
        """
        batch_size, num_classes = output.shape

        # Seuil pour binariser les prédictions (0 ou 1)
        output_pred = (output > 0.5).float()  # Matrice binaire (batch_size, num_classes)

        # Calcul des activations des super-classes via la matrice H (Hiérarchie)
        H = self.H.float()  # Matrice hiérarchique (num_classes, num_classes)
        Hs = H @ output_pred.T  # (num_classes, batch_size), prédit les activations correctes des super-classes
        
        parents = torch.sum(H, dim=1)
        parents_batch = parents.repeat(batch_size,1).T  # (batch_size, num_classes)

        # Vérifier que si une classe est activée, sa super-classe l'est aussi
        # Si une classe est activée mais pas sa super-classe, cela viole la règle
        violation_mask = (output_pred.T > 0) & (Hs == 0) & (parents_batch != 0)  # (num_classes, batch_size)
        
        # Compter les échantillons respectant la règle (aucune violation)
        batch_respect = (torch.sum(violation_mask, dim=0) == 0).float()  # (batch_size,)

        # Calcul du pourcentage d'échantillons respectant la C-Rule
        total_respect = torch.mean(batch_respect)
        self.metrics[MetricsLabels.d_rule_respect] = total_respect



    def  e_rule_respect_percentage(self, output: torch.Tensor, target, label_matrix: torch.Tensor):
        """
        Calcule le pourcentage d'échantillons respectant la E-Rule.

        Args:
            output (torch.Tensor): Matrice des prédictions du modèle (batch_size, num_classes).
            label_matrix (torch.Tensor): Matrice one-hot des labels réels (batch_size, num_classes).

        Returns:
            float: Pourcentage des échantillons respectant la E-Rule.
        """
        output_pred = (output > 0.5).float()  # Matrice binaire (batch_size, num_classes)

        H = self.H.float()  # Matrice hiérarchique (num_classes, num_classes)
        Hs = H @ output_pred.T

        # Vérifier que si une classe est activée, sa super-classe l'est aussi
        # Si une classe est activée mais pas sa super-classe, cela viole la règle
        violation_mask = (Hs > 1)  # (num_classes, batch_size)
        
        # Compter les échantillons respectant la règle (aucune violation)
        batch_respect = (torch.sum(violation_mask, dim=0) == 0).float()  # (batch_size,)

        # Calcul du pourcentage d'échantillons respectant la C-Rule
        total_respect = torch.mean(batch_respect)
        self.metrics[MetricsLabels.e_rule_respect] = total_respect



    def accuracy_topk_1_5(self, output, target, label_matrix):
        """
        Calcule la précision top-1 et top-5 pour la segmentation logique.

        Args:
            output (torch.Tensor): Prédictions du modèle (logits).
            target (torch.Tensor): Labels réels.
            label_matrix (torch.Tensor): Matrice des labels.

        """
        probas_branches_input = get_logicseg_predictions(output, label_matrix)
        probas_branches_target = get_logicseg_predictions(target, label_matrix)

        # Top-1 Accuracy
        _, indices_branches_in = probas_branches_input.topk(1, dim=1)
        _, indices_branches_target = probas_branches_target.topk(1, dim=1)
        acc1 = (indices_branches_in == indices_branches_target).sum().item() / indices_branches_in.size(0)

        # Top-5 Accuracy
        _, indices_branches_in = probas_branches_input.topk(5, dim=1)
        _, indices_branches_target = probas_branches_target.topk(5, dim=1)
        acc5 = (indices_branches_in == indices_branches_target).sum().item() / indices_branches_in.size(0)

        # Stocker les résultats
        self.metrics[MetricsLabels.accuracy_top1] = acc1
        self.metrics[MetricsLabels.accuracy_top5] = acc5



    def reset_metrics(self):
        """Réinitialise les métriques stockées."""
        self.metrics = {key: -1 for key in self.metrics}


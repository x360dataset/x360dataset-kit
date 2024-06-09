from torch import nn

class BCEloss(nn.Module):
    def __init__(self):
        super(BCEloss, self).__init__()
        self.CE = nn.CrossEntropyLoss()
        # self.MSE = nn.MSELoss()

    def forward(self, pred, target, aux_pred):
        CE_score = self.CE(pred, target)
        return CE_score


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
from scipy.stats import rankdata
import random


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def permutation_loss(features, target_order):
    """
    Computes the permutation loss between predicted permutation and target permutation of features.

    Args:
        features (torch.Tensor): The input features, shape [num_frames, feature_dim].
        target_permutation (torch.Tensor): The target permutation indices, shape [num_frames].

    Returns:
        torch.Tensor: The permutation loss.

    """
    features = features / torch.max(torch.abs(features))
    # [0, 1, 2]
    # target_order = torch.Tensor(target_order).to(features.device)

    t = target_order.shape[0]

    # Compute the target permutation matrix
    pairwise_disagreements = torch.zeros(t).to(DEVICE)
    order_matrix = torch.zeros(t).to(DEVICE)

    idx = 0
    for i in range(t):
        for j in range(i + 1, t):
            order_matrix[idx] = target_order[i] - target_order[j]  # >?

            pairwise_disagreements[idx] = torch.sum(features[i] - features[j])  # >?
            idx += 1

    order_matrix = order_matrix / (torch.max(torch.abs(order_matrix)) + 1e-8)
    pairwise_disagreements = pairwise_disagreements / \
                             (torch.max(torch.abs(pairwise_disagreements)) + 1e-8)

    # print("pairwise_disagreements:", pairwise_disagreements)

    # print("order loss:", order_matrix)

    normalized_loss = torch.sum(torch.abs(order_matrix - pairwise_disagreements))

    return normalized_loss


def batch_permutation_loss(features, target_order):
    """
    Args:
        features (torch.Tensor): The input features, shape [batch_size, num_frames, feature_dim].
        target_permutation (torch.Tensor): The target permutation indices, shape [batch_size, num_frames].

    """
    for idx, data in enumerate(zip(features, target_order)):

        l = permutation_loss(data[0], data[1])
        if idx == 0:
            loss= l
        else:
            loss += l

    return loss




def contrastive_loss(features1, features2, margin=5.0, inter_set_weight=0.5,
                     intra_set_weight=1, return_both=False):
    """
    Computes the contrastive loss between two sets of features.

    Args:
        features1 (torch.Tensor): The first set of features, shape [batch_size, feature_dim].
        features2 (torch.Tensor): The second set of features, shape [batch_size, feature_dim].
        margin (float): The margin value for the contrastive loss.

    Returns:
        torch.Tensor: The contrastive loss.

    """
    # Compute pairwise Euclidean distances between features
    features1 = features1 / torch.max(torch.abs(features1))
    features2 = features2 / torch.max(torch.abs(features2))


    inter_distances = torch.cdist(features1, features2, p=2)  # p-norm distance
    intra_distances = torch.cdist(features1, features1, p=2)


    # print("intra_distances:", intra_distances)
    #
    # print("inter_distances:", inter_distances)


    # Compute positive pair loss
    pull_loss = torch.mean(intra_distances)  # pair to pair distance

    # Compute intra-set loss
    # Similarity

    repel_loss = torch.clamp(margin - inter_distances, min=0).mean()

    # print("pull_loss:", pull_loss)
    # print("repel_loss:", repel_loss)


    # Total contrastive loss
    loss = inter_set_weight * (pull_loss + repel_loss)

    if return_both:
        return repel_loss.detach().cpu().numpy(), pull_loss.detach().cpu().numpy()

    return loss.mean()


# def view_
def get_order_loss(model, imgs, time_window=3):
    b, t, c, h, w = imgs.shape
    # removed "Batch"
    # ordered_klgs = klgs[0, shuffled_indices]

    for i in range(b):
        shuffled_indices = random.sample(range(t), k=time_window)
        ranked_indices = rankdata(shuffled_indices)

        ordered_imgs = imgs[i, shuffled_indices]

        feats = model.get_feature(ordered_imgs)
        if i == 0:
            order_loss = permutation_loss(feats, ranked_indices)
        else:
            order_loss += permutation_loss(feats, ranked_indices)

    return order_loss / b


def get_contrastive_loss(model, imgsA, imgsB):
    # print("imgs.shape:", imgs.shape)

    for i in range(imgsA.shape[0]):
        featsA = model.get_feature(imgsA[i])
        featsB = model.get_feature(imgsB[i])

        # In Order
        if i == 0:
            loss = contrastive_loss(featsA, featsB)
        else:
            loss += contrastive_loss(featsA, featsB)

    return loss


def get_contrastive_loss_validate(model, imgsA, imgsB):
    for i in range(imgsA.shape[0]):
        featsA = model.get_feature(imgsA[i])
        featsB = model.get_feature(imgsB[i])

        if i == 0:
            inter_loss, intra_loss = contrastive_loss(featsA, featsB, return_both=True)
        else:
            loss_cache = contrastive_loss(featsA, featsB, return_both=True)
            inter_loss += loss_cache[0]
            intra_loss += loss_cache[1]
        # print("inter_loss:", inter_loss, ", intra_loss:", intra_loss)
    return inter_loss, intra_loss

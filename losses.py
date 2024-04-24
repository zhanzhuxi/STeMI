import torch
from torch.nn import functional as F


def calc_cls_loss(pred: torch.Tensor, test: torch.Tensor) -> torch.Tensor:
    test = test.type(torch.long)
    num_pos = test.sum()
    pred = pred.unsqueeze(-1)
    pred = torch.cat([1 - pred, pred], dim=-1)
    loss = focal_loss(pred, test)
    loss = loss / num_pos
    return loss


def iou_offset(offset_a: torch.Tensor, offset_b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    left_a, right_a = offset_a[:, 0], offset_a[:, 1]
    left_b, right_b = offset_b[:, 0], offset_b[:, 1]

    length_a = left_a + right_a
    length_b = left_b + right_b

    intersect = torch.min(left_a, left_b) + torch.min(right_a, right_b)
    intersect[intersect < 0] = 0
    union = length_a + length_b - intersect
    union[union <= 0] = eps

    iou = intersect / union
    return iou


def calc_loc_loss(pred_loc: torch.Tensor, test_loc: torch.Tensor, cls_label: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    cls_label = cls_label.type(torch.bool)
    pred_loc = pred_loc[cls_label]
    test_loc = test_loc[cls_label]
    iou = iou_offset(pred_loc, test_loc)
    loss = -torch.log(iou + eps).mean()
    return loss


def calc_ctr_loss(pred, test, pos_mask):
    pos_mask = pos_mask.type(torch.bool)
    pred = pred[pos_mask]
    test = test[pos_mask]
    loss = F.binary_cross_entropy(pred, test)
    return loss


def one_hot_embedding(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    eye = torch.eye(num_classes, device=labels.device)
    return eye[labels]


def focal_loss(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.25, gamma: float = 2) -> torch.Tensor:
    _, num_classes = x.shape
    t = one_hot_embedding(y, num_classes)
    # p_t = p if t > 0 else 1-p
    p_t = x * t + (1 - x) * (1 - t)
    # alpha_t = alpha if t > 0 else 1-alpha
    alpha_t = alpha * t + (1 - alpha) * (1 - t)
    # FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    fl = -alpha_t * (1 - p_t).pow(gamma) * p_t.log()
    fl = fl.sum()
    return fl


def reconstruction_loss(reconstructed_feature, original_feature):
    loss = F.l1_loss(reconstructed_feature, original_feature)
    return loss


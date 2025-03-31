from loss.cross_entropy_loss import CrossEntropyLoss
from loss.triplet_loss import TripletLoss
from loss.center_loss import CenterLoss
from loss.arcface import ArcFaceLoss

class CombinedLoss:
    def __init__(self, loss_fns):
        self.loss_fns = loss_fns

    def __call__(self, outputs, targets, features=None):
        total_loss = 0
        for loss_fn in self.loss_fns:
            if isinstance(loss_fn, (TripletLoss, CenterLoss, ArcFaceLoss)):
                total_loss += loss_fn(features, targets)
            else:
                total_loss += loss_fn(outputs, targets)
        return total_loss


def build_loss(loss_list, num_classes=None, feat_dim=None):
    loss_fns = []

    if "cross_entropy" in loss_list:
        loss_fns.append(CrossEntropyLoss())

    if "triplet" in loss_list:
        loss_fns.append(TripletLoss(margin=0.3))

    if "center" in loss_list:
        loss_fns.append(CenterLoss(num_classes=num_classes, feat_dim=feat_dim))

    if "arcface" in loss_list:
        loss_fns.append(ArcFaceLoss(feat_dim=feat_dim, num_classes=num_classes))

    return CombinedLoss(loss_fns)

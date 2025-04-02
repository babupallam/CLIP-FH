from loss.cross_entropy_loss import CrossEntropyLoss
from loss.triplet_loss import TripletLoss
from loss.center_loss import CenterLoss
from loss.arcface import ArcFaceLoss
from loss.contrastive_loss import supcon_loss, clip_contrastive_loss


class CombinedLoss:
    def __init__(self, loss_fns, contrastive=None):
        self.loss_fns = loss_fns
        self.contrastive = contrastive  # Add support for supcon/clip

    def __call__(self, outputs=None, targets=None, features=None, text_features=None, mode="contrastive"):
        if mode == "contrastive" and self.contrastive is not None:
            return self.contrastive(features, text_features, targets)

        total_loss = 0
        for loss_fn in self.loss_fns:
            if isinstance(loss_fn, (TripletLoss, CenterLoss, ArcFaceLoss)):
                total_loss += loss_fn(features, targets)
            else:
                total_loss += loss_fn(outputs, targets)
        return total_loss


def build_loss(loss_list, num_classes=None, feat_dim=None):
    loss_fns = []
    contrastive_fn = None

    if "cross_entropy" in loss_list:
        loss_fns.append(CrossEntropyLoss())

    if "triplet" in loss_list:
        loss_fns.append(TripletLoss(margin=0.3))

    if "center" in loss_list:
        loss_fns.append(CenterLoss(num_classes=num_classes, feat_dim=feat_dim))

    if "arcface" in loss_list:
        loss_fns.append(ArcFaceLoss(feat_dim=feat_dim, num_classes=num_classes))

    if "supcon" in loss_list:
        contrastive_fn = supcon_loss

    if "clip" in loss_list:
        contrastive_fn = clip_contrastive_loss

    return CombinedLoss(loss_fns, contrastive=contrastive_fn)

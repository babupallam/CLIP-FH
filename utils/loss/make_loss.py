from utils.loss.cross_entropy_loss import CrossEntropyLoss
from utils.loss.triplet_loss import TripletLoss
from utils.loss.center_loss import CenterLoss
from utils.loss.supcon import SupConLoss
from utils.loss.arcface import ArcFace

class CombinedLoss:
    def __init__(self, loss_fns, contrastive=None):
        self.loss_fns = loss_fns
        self.contrastive = contrastive  # Add support for supcon/clip

    def __call__(self, features=None, text_features=None, targets=None, mode="contrastive"):
        if mode == "contrastive" and self.contrastive is not None:
            #print(f"🧪 Calling contrastive loss with:\n  text_feats: {text_features.shape}\n  img_feats: {features.shape}\n  labels: {targets.shape}")
            return self.contrastive(text_features, features, targets, targets)

        total_loss = 0
        for loss_fn in self.loss_fns:
            if isinstance(loss_fn, (TripletLoss, CenterLoss, ArcFace)):
                total_loss += loss_fn(features, targets)
            else:
                total_loss += loss_fn(features, targets)
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
        loss_fns.append(ArcFace(feat_dim=feat_dim, num_classes=num_classes))

    if "supcon" in loss_list:
        contrastive_fn = SupConLoss(device="cuda")

    #print(f"[make_loss] Using contrastive_fn = {contrastive_fn}")

    return CombinedLoss(loss_fns, contrastive=contrastive_fn)

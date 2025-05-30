from utils.loss.cross_entropy_loss import CrossEntropyLoss
from utils.loss.triplet_loss import TripletLoss
from utils.loss.center_loss import CenterLoss
from utils.loss.supcon import SupConLoss
from utils.loss.arcface import ArcFace

# This class combines multiple loss functions into one
class CombinedLoss:
    def __init__(self, loss_fns, contrastive=None):
        self.loss_fns = loss_fns
        self.contrastive = contrastive  # Used for contrastive losses like SupCon or CLIP

    def __call__(self, features=None, text_features=None, targets=None, mode="contrastive"):
        # If mode is contrastive and a contrastive loss function is set, use it
        if mode == "contrastive" and self.contrastive is not None:
            # Call contrastive loss with text and image features
            return self.contrastive(text_features, features, targets, targets)

        total_loss = 0
        # For other loss functions, compute each loss and add to total
        for loss_fn in self.loss_fns:
            # These losses require features and targets
            if isinstance(loss_fn, (TripletLoss, CenterLoss, ArcFace)):
                total_loss += loss_fn(features, targets)
            else:
                # For others like CrossEntropy
                total_loss += loss_fn(features, targets)
        return total_loss

# This function builds and returns a CombinedLoss object
# based on the loss types given in loss_list
def build_loss(loss_list, num_classes=None, feat_dim=None):
    loss_fns = []
    contrastive_fn = None

    # Add cross entropy loss if specified
    if "cross_entropy" in loss_list:
        loss_fns.append(CrossEntropyLoss())

    # Add triplet loss if specified
    if "triplet" in loss_list:
        loss_fns.append(TripletLoss(margin=0.3))

    # Add center loss if specified
    if "center" in loss_list:
        loss_fns.append(CenterLoss(num_classes=num_classes, feat_dim=feat_dim))

    # Add ArcFace loss if specified
    if "arcface" in loss_list:
        loss_fns.append(ArcFace(feat_dim=feat_dim, num_classes=num_classes))

    # Add SupCon contrastive loss if specified
    if "supcon" in loss_list:
        contrastive_fn = SupConLoss(device="cuda")

    # Return the combined loss module
    return CombinedLoss(loss_fns, contrastive=contrastive_fn)

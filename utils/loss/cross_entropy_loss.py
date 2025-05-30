import torch.nn as nn

# This class wraps PyTorch's built-in CrossEntropyLoss
# It is used for classification problems
class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

        # Create the standard cross-entropy loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        # Compute the loss between model outputs and true labels
        return self.loss_fn(outputs, targets)

import torch                              # PyTorch for tensor operations and model handling
import torch.nn as nn                     # Neural network components like layers and loss functions
import torch.optim as optim               # Optimizers (Adam, SGD, etc.)
from tqdm import tqdm                     # Progress bar for iterating over data
import os                                 # OS module for file and directory operations


class FinetuneTrainerStage1:
    """
    Stage 1 fine-tuning trainer for CLIP-based classification.

    Description:
    - This stage fine-tunes CLIP's image encoder.
    - A linear classifier is trained on top of the image features.
    - The text encoder remains frozen (unused).
    """

    def __init__(self, clip_model, classifier, train_loader, config, device):
        """
        Initialize training setup.

        Args:
            clip_model (CLIP): Preloaded CLIP model (text encoder frozen).
            classifier (nn.Linear): A linear layer to classify extracted image features.
            train_loader (DataLoader): DataLoader yielding batches of (image, label) pairs.
            config (dict): Contains training hyperparameters and paths:
                - "epochs": Number of training epochs
                - "lr": Learning rate
                - "save_path": Path to save model weights after training
                - "log_path": Path to save training log
            device (torch.device): Target computation device ("cuda" or "cpu")
        """
        self.clip_model = clip_model                          # Full CLIP model (only image encoder will be used)
        self.classifier = classifier.to(device)               # Classifier head for classification task
        self.train_loader = train_loader                      # DataLoader providing training samples
        self.device = device                                  # CUDA or CPU device

        # Load training hyperparameters from config
        self.epochs = config["epochs"]
        self.lr = config["lr"]
        self.save_path = config["save_path"]
        self.log_path = config["log_path"]

        # Define classification loss
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer includes both the classifier and the CLIP image encoder (visual) parameters
        self.optimizer = optim.Adam(
            list(self.clip_model.visual.parameters()) + list(self.classifier.parameters()),
            lr=self.lr
        )

    def train(self):
        """
        Fine-tune the image encoder and classifier on the training set.

        - The CLIP image encoder (encode_image) is updated via gradient descent.
        - The CLIP text encoder is frozen and unused.
        - Outputs logs and saves model after training.
        """
        self.clip_model.train()                               # Set model to training mode
        log = []                                               # List to collect training logs

        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0                                   # Total loss for epoch
            total = 0                                          # Total number of samples
            correct = 0                                        # Count of correctly predicted samples

            # Loop over training batches
            for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()                    # Clear gradients

                # 🔓 UNFREEZE IMAGE ENCODER — compute gradients for visual features
                features = self.clip_model.encode_image(images)  # Forward pass through image encoder

                # Forward pass through classifier
                outputs = self.classifier(features)

                # Compute loss between predicted class scores and true labels
                loss = self.criterion(outputs, labels)

                # Backpropagate the loss
                loss.backward()

                # Update parameters of image encoder and classifier
                self.optimizer.step()

                # Track metrics
                total_loss += loss.item()
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()

            # Compute accuracy and average loss for the epoch
            acc = 100.0 * correct / total
            avg_loss = total_loss / len(self.train_loader)
            log_line = f"Epoch {epoch}: Loss={avg_loss:.4f}, Acc={acc:.2f}%"
            print(log_line)
            log.append(log_line)

        # Save trained image encoder weights (entire CLIP model's state_dict)
        torch.save(self.clip_model.state_dict(), self.save_path)
        print(f"\n✅ Model saved to: {self.save_path}")

        # Save training logs to file
        log_dir = os.path.dirname(self.log_path)
        os.makedirs(log_dir, exist_ok=True)
        with open(self.log_path, "w") as f:
            f.write("\n".join(log))
        print(f"📝 Training log saved to: {self.log_path}")

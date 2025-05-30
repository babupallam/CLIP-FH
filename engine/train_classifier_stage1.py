# Standard library imports for file operations and system functions
import os
# Standard library import for time-related functions (to measure epoch duration, etc.)
import time

# PyTorch imports for creating and training neural networks
import torch
import torch.nn as nn
import torch.optim as optim

# tqdm provides a progress bar for loops
from tqdm import tqdm

# Utility imports for validation, logging, naming outputs, saving/loading models, and loss functions
from utils.eval_utils import validate
from utils.logger import setup_logger
from utils.naming import build_filename
from utils.save_load_models import save_checkpoint
from utils.loss.cross_entropy_loss import CrossEntropyLoss
from utils.loss.triplet_loss import TripletLoss
from utils.loss.arcface import ArcFace

import torch.nn.functional as F


class FinetuneTrainerStage1:
    """
    Fine-tunes the image encoder of CLIP using a linear classifier,
    with the text encoder frozen.
    """
    def __init__(self, clip_model, classifier, train_loader, val_loader, config, device):
        # Store the CLIP model and classifier; move the classifier to the device (GPU or CPU)
        self.clip_model = clip_model
        self.classifier = classifier.to(device) # Move classifier to the specified device (GPU or CPU).
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = config["epochs"] # Number of training epochs.
        self.lr = config["lr"] # Learning rate for the optimizer.
        self.save_path = config["save_path"] # Path to save model checkpoints.
        self.log_path = config["log_path"] # Path to save training logs.
        self.config = config

        # Define the training losses (cross-entropy and triplet)
        self.ce_loss = CrossEntropyLoss()
        self.triplet_loss = TripletLoss(margin=0.3)

        # Early stopping patience: number of epochs with no improvement before stopping
        self.early_stop_patience = config.get("early_stop_patience", 3)
        self.no_improve_epochs = 0

        # Setup optimizer with different learning rates for the image encoder and classifier
        # (Image encoder uses self.lr, classifier uses self.lr * 0.1)
        self.optimizer = optim.Adam([
            {"params": clip_model.visual.parameters(), "lr": self.lr},
            {"params": classifier.parameters(), "lr": self.lr * 0.1}
        ], weight_decay=1e-4)

        # Setup logging by creating the log directory if it doesn't exist
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True) # Create log directory if it doesn't exist.
        self.logger = setup_logger(self.log_path) # Initialize logger.



    def train(self):
        """
        Runs the training loop for frozen-text classifier fine-tuning.
        """
        # Set model to training mode (affects layers like dropout, batch normalization, etc.)
        self.clip_model.train()
        best_acc1 = 0.0 # Track the best validation Rank-1 accuracy

        # Main loop over the total number of epochs
        for epoch in range(1, self.epochs + 1): # Iterate over epochs.
            start_time = time.time() # Mark the start time for this epoch
            total_loss, total = 0.0, 0
            correct_rank1 = correct_rank5 = correct_rank10 = 0

            # Iterate over the training data by batches
            for batch_idx, (images, labels) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} Training")): # Iterate over batches.
                images, labels = images.to(self.device), labels.to(self.device) # Move images and labels to device.
                self.optimizer.zero_grad() # Reset gradients to zero before computing new ones.

                # Encode images using CLIP's visual encoder
                features = self.clip_model.encode_image(images).float() # Get feature vectors from the image encoder.
                # Run the classifier on the extracted features
                features = F.normalize(features, dim=1)  # Normalize before classifier --- from v4

                #  This ensures ArcFace gets the label input it requires.
                if self.config.get("classifier", "linear") == "arcface":
                    outputs = self.classifier(features, labels)
                else:
                    outputs = self.classifier(features)

                # Check for numerical issues in outputs
                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    self.logger.warning(
                        f"[WARNING] Detected NaNs or Infs in logits at epoch {epoch}, batch {batch_idx}")

                # Calculate cross-entropy loss
                ce = self.ce_loss(outputs, labels)
                # Calculate triplet loss (wrapped in a try-except, as it might fail if there aren't enough valid samples)
                try:
                    tri = self.triplet_loss(features, labels)
                except Exception as e:
                    self.logger.warning(f"[TripletLoss Error] {e}")
                    tri = torch.tensor(0.0, device=self.device)

                # Final training loss (sum of cross-entropy and triplet losses)
                loss = ce + tri

                # Backpropagation: compute gradients
                loss.backward()
                # Check for any NaNs in the loss
                if torch.isnan(loss):
                    self.logger.warning(f"[WARNING] Loss is NaN at epoch {epoch}, batch {batch_idx}")
                    continue

                # Clip gradients to avoid exploding gradients
                #v2 - disabled this explode gradient solution
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.clip_model.visual.parameters(), max_norm=1.0)

                # Update model parameters
                self.optimizer.step()

                # Accumulate training loss
                total_loss += loss.item()
                # Gather top-k predictions to measure accuracy at different ranks
                _, pred_rankk = outputs.topk(10, dim=1)
                # Increase count of total samples
                total += labels.size(0)
                # Check how many predictions match the label at rank 1, 5, and 10
                correct_rank1 += (pred_rankk[:, :1] == labels.unsqueeze(1)).sum().item()
                correct_rank5 += (pred_rankk[:, :5] == labels.unsqueeze(1)).sum().item()
                correct_rank10 += (pred_rankk[:, :10] == labels.unsqueeze(1)).sum().item()

                # Log debug info for the first batch in each epoch
                if batch_idx == 0: # Log debug information for the first batch.
                    self.logger.info(f"[DEBUG] Batch 0 - image shape: {images.shape}, labels: {labels.shape}")
                    self.logger.info(f"[DEBUG] img_features: {features.shape}, prompt: N/A, pooled: {outputs.shape}")
                    self.logger.info(f"[DEBUG] logits std: {outputs.std().item():.4f}")
                    self.logger.info(f"[DEBUG] CE loss: {ce.item():.4f}, Triplet loss: {tri.item():.4f}, Total: {loss.item():.4f}")

            # Compute average training loss for the entire epoch
            avg_loss = total_loss / len(self.train_loader)
            # Compute rank 1, 5, and 10 accuracy
            acc1 = 100.0 * correct_rank1 / total
            acc5 = 100.0 * correct_rank5 / total
            acc10 = 100.0 * correct_rank10 / total
            # Measure how long this epoch took
            epoch_time = time.time() - start_time

            # Log the training results for the epoch
            self.logger.info(f"[Epoch {epoch}] Total Loss={avg_loss:.4f}")
            self.logger.info(f"[Epoch {epoch}] Acc@1={acc1:.2f}% | Acc@5={acc5:.2f}% | Acc@10={acc10:.2f}%")

            # === Validation ===
            # Attach the classifier to the clip model to use the unified interface in the validate function
            self.clip_model.classifier = self.classifier  # Required for validate()

            # Run the validation function (ReID-type validation)
            val_metrics = validate(
                model=self.clip_model,
                prompt_learner=None,
                val_loader=self.val_loader,
                device=self.device,
                log=self.logger.info,
                val_type="reid",  # <--  classifier not now
                batch_size=self.config.get("batch_size", 32),
                loss_fn=[self.ce_loss, self.triplet_loss]  # pass both losses unchanged
            )

            # Check if this is the best validation result so far (based on rank1)
            if epoch == 1 or val_metrics['rank1'] > best_acc1:
                best_acc1 = val_metrics['rank1']
                self.no_improve_epochs = 0  # Reset the no-improvement counter
                model_name = build_filename(self.config, epoch, stage="image", extension="_BEST.pth", timestamped=False)
                best_model_path = os.path.join(self.config['save_dir'], model_name)
                # Save the current best model checkpoint
                save_checkpoint(
                    model=self.clip_model,
                    classifier=self.classifier,
                    optimizer=self.optimizer,
                    config=self.config,
                    epoch=epoch,
                    val_metrics=val_metrics,
                    path=best_model_path,
                    is_best=True,
                    scheduler=getattr(self, "scheduler", None)
                )
                self.logger.info(f"Saving best model at epoch {epoch} -> {best_model_path}")
            else:
                # If validation didn't improve, increment the no-improvement counter
                self.no_improve_epochs += 1
                self.logger.info(
                    f"[INFO] No improvement in Rank-1 ({val_metrics['rank1']:.2f}%), patience = {self.no_improve_epochs}/{self.early_stop_patience}")

            # If we've hit the limit of no improvement, stop early
            if self.no_improve_epochs >= self.early_stop_patience:
                self.logger.info(
                    f"[EARLY STOPPING] No improvement for {self.early_stop_patience} consecutive epochs. Stopping at epoch {epoch}.")
                break


        # === Final model save ===
        # Build a filename to save the final model of this training session
        model_name = build_filename(self.config, epoch, stage="image", extension="_FINAL.pth", timestamped=False)
        final_model_path = os.path.join(self.config['save_dir'], model_name)
        # Save the final model checkpoint
        save_checkpoint(
            model=self.clip_model,
            classifier=self.classifier,
            optimizer=self.optimizer,
            config=self.config,
            epoch=epoch,
            val_metrics=val_metrics,
            path=final_model_path,
            is_best=False,
            scheduler=getattr(self, "scheduler", None)
        )
        self.logger.info(f"Model saved to: {final_model_path}")

import os
import csv
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils.eval_utils import validate
from utils.logger import setup_logger
from utils.naming import build_filename
from utils.save_load_models import save_checkpoint
from utils.loss.cross_entropy_loss import CrossEntropyLoss
from utils.loss.triplet_loss import TripletLoss
from utils.loss.arcface import ArcFace



class FinetuneTrainerStage1:
    """
    Fine-tunes the image encoder of CLIP using a linear classifier,
    with the text encoder frozen.
    """
    def __init__(self, clip_model, classifier, train_loader, val_loader, config, device):
        self.clip_model = clip_model
        self.classifier = classifier.to(device) # Move classifier to the specified device (GPU or CPU).
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = config["epochs"] # Number of training epochs.
        self.lr = config["lr"] # Learning rate for the optimizer.
        self.save_path = config["save_path"] # Path to save model checkpoints.
        self.log_path = config["log_path"] # Path to save training logs.
        self.csv_path = self.log_path.replace(".txt", ".csv") # Path to save training logs in CSV format.
        self.config = config

        # Define training components
        self.ce_loss = CrossEntropyLoss()
        self.triplet_loss = TripletLoss(margin=0.3)

        #self.optimizer = optim.Adam(
        #    list(self.clip_model.visual.parameters()) + list(self.classifier.parameters()), lr=self.lr
        #) # Optimizer for updating model weights. Only image encoder and classifier are updated.
        self.optimizer = optim.Adam([
            {"params": clip_model.visual.parameters(), "lr": self.lr},
            {"params": classifier.parameters(), "lr": self.lr * 0.1}
        ], weight_decay=1e-4)

        # Setup logging
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True) # Create log directory if it doesn't exist.
        self.logger = setup_logger(self.log_path) # Initialize logger.

        # Prepare CSV log
        with open(self.csv_path, "w", newline="") as f: # Open CSV file for writing.
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "train_acc1", "train_acc5", "train_acc10",
                "val_rank1", "val_rank5", "val_rank10", "val_mAP",
                "learning_rate", "epoch_time_sec"
            ])

    def train(self):
        """
        Runs the training loop for frozen-text classifier fine-tuning.
        """
        self.clip_model.train() # Set the model to training mode.
        best_acc1 = 0.0 # Initialize best validation accuracy.

        for epoch in range(1, self.epochs + 1): # Iterate over epochs.
            start_time = time.time() # Start time for epoch.
            total_loss, total = 0.0, 0
            correct_rank1 = correct_rank5 = correct_rank10 = 0

            for batch_idx, (images, labels) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} Training")): # Iterate over batches.
                images, labels = images.to(self.device), labels.to(self.device) # Move images and labels to device.
                self.optimizer.zero_grad() # Zero gradients.

                # Encode images using CLIP visual encoder and classify
                features = self.clip_model.encode_image(images) # Encode images using the image encoder.
                outputs = self.classifier(features) # Classify the encoded features.

                if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                    self.logger.warning(
                        f"[WARNING] Detected NaNs or Infs in logits at epoch {epoch}, batch {batch_idx}")

                ce = self.ce_loss(outputs, labels)
                try:
                    tri = self.triplet_loss(features, labels)
                except Exception as e:
                    self.logger.warning(f"[TripletLoss Error] {e}")
                    tri = torch.tensor(0.0, device=self.device)

                loss = ce + tri # calculate loss (Cross Entropy + Triplet Loss (Classic hybrid))

                loss.backward() # Backpropagate loss.
                if torch.isnan(loss):
                    self.logger.warning(f"[WARNING] Loss is NaN at epoch {epoch}, batch {batch_idx}")
                    continue

                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.clip_model.visual.parameters(), max_norm=1.0)
                self.optimizer.step() # Update model weights.

                # Accumulate metrics
                total_loss += loss.item() # Accumulate loss.
                _, pred_rankk = outputs.topk(10, dim=1) # Get top 10 predictions.
                total += labels.size(0) # Accumulate total number of samples.
                correct_rank1 += (pred_rankk[:, :1] == labels.unsqueeze(1)).sum().item() # Accumulate correct predictions at rank 1.
                correct_rank5 += (pred_rankk[:, :5] == labels.unsqueeze(1)).sum().item() # Accumulate correct predictions at rank 5.
                correct_rank10 += (pred_rankk[:, :10] == labels.unsqueeze(1)).sum().item() # Accumulate correct predictions at rank 10.

                if batch_idx == 0: # Log debug information for the first batch.
                    self.logger.info(f"[DEBUG] Batch 0 - image shape: {images.shape}, labels: {labels.shape}")
                    self.logger.info(f"[DEBUG] img_features: {features.shape}, prompt: N/A, pooled: {outputs.shape}")
                    self.logger.info(f"[DEBUG] logits std: {outputs.std().item():.4f}")
                    self.logger.info(f"[DEBUG] CE loss: {ce.item():.4f}, Triplet loss: {tri.item():.4f}, Total: {loss.item():.4f}")

            # Epoch-level metrics
            avg_loss = total_loss / len(self.train_loader) # Calculate average loss.
            acc1 = 100.0 * correct_rank1 / total # Calculate accuracy at rank 1.
            acc5 = 100.0 * correct_rank5 / total # Calculate accuracy at rank 5.
            acc10 = 100.0 * correct_rank10 / total # Calculate accuracy at rank 10.
            epoch_time = time.time() - start_time # Calculate epoch time.

            self.logger.info(f"[Epoch {epoch}] Total Loss={avg_loss:.4f}")
            self.logger.info(f"[Epoch {epoch}] Acc@1={acc1:.2f}% | Acc@5={acc5:.2f}% | Acc@10={acc10:.2f}%")

            # === Validation ===
            self.clip_model.classifier = self.classifier  # Required for validate()

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

            # === Save best checkpoint ===
            if epoch == 1 or val_metrics['rank1'] > best_acc1: # Save model if it's the first epoch or best validation accuracy.
                best_acc1 = val_metrics['rank1']
                model_name = build_filename(self.config, epoch, stage="image", extension="_BEST.pth", timestamped=False) # Build model filename.
                best_model_path = os.path.join(self.config['save_dir'], model_name) # Build model save path.
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
                ) # Save best model checkpoint.
                self.logger.info(f"Saving best model at epoch {epoch} -> {best_model_path}")
            else:
                self.logger.info(f"[INFO] No improvement in Rank-1 ({val_metrics['rank1']:.2f}%), skipping checkpoint.")

            # === Log to CSV ===
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch,
                    avg_loss,
                    acc1,
                    acc5,
                    acc10,
                    val_metrics.get('rank1', 0.0),
                    val_metrics.get('rank5', 0.0),
                    val_metrics.get('rank10', 0.0),
                    val_metrics.get('mAP', 0.0),
                    self.lr,
                    epoch_time
                ])

        # === Final model save ===
        model_name = build_filename(self.config, epoch, stage="image", extension="_FINAL.pth", timestamped=False) # Build final model filename.
        final_model_path = os.path.join(self.config['save_dir'], model_name) # Build final model save path.
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
        ) # Save final model checkpoint.
        self.logger.info(f"Model saved to: {final_model_path}")
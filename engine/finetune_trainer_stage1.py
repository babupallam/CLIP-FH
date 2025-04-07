import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import logging
import time
import csv


class FinetuneTrainerStage1:
    def __init__(self, clip_model, classifier, train_loader, config, device):
        self.clip_model = clip_model
        self.classifier = classifier.to(device)
        self.train_loader = train_loader
        self.device = device

        # Hyperparameters
        self.epochs = config["epochs"]
        self.lr = config["lr"]
        self.save_path = config["save_path"]
        self.log_path = config["log_path"]
        self.csv_path = self.log_path.replace(".txt", ".csv")

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            list(self.clip_model.visual.parameters()) + list(self.classifier.parameters()),
            lr=self.lr
        )

        # Setup logging
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        logging.basicConfig(
            filename=self.log_path,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filemode="w"
        )
        self.logger = logging.getLogger()
        self.logger.addHandler(logging.StreamHandler())  # Also log to console

        # Setup CSV
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "loss", "accuracy", "learning_rate", "epoch_time_sec", "num_samples"])

    def train(self):
        self.clip_model.train()
        total_samples = len(self.train_loader.dataset)
        self.logger.info(f"Starting training on {total_samples} samples for {self.epochs} epochs")

        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            total_loss = 0.0
            total = 0
            correct = 0

            for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                features = self.clip_model.encode_image(images)
                outputs = self.classifier(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()

            avg_loss = total_loss / len(self.train_loader)
            acc = 100.0 * correct / total
            epoch_time = time.time() - start_time

            # Log results
            log_line = f"Epoch {epoch}: Loss={avg_loss:.4f}, Acc={acc:.2f}%, Time={epoch_time:.2f}s, LR={self.lr}"
            self.logger.info(log_line)

            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, avg_loss, acc, self.lr, epoch_time, total])

        # Save final model
        torch.save(self.clip_model.state_dict(), self.save_path)
        self.logger.info(f"Model saved to: {self.save_path}")
        self.logger.info(f"Training metrics logged to: {self.csv_path}")

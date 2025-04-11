import sys                                # System-specific parameters and functions
import os                                 # OS module for path and environment management

# 🔧 Add the project root directory to PYTHONPATH to ensure internal imports work correctly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if "datasets" not in os.listdir(PROJECT_ROOT):
    raise RuntimeError("PROJECT_ROOT is misaligned. Check the relative path in the script.")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import time
import csv
import logging
from tqdm import tqdm

from datasets.build_dataloader import get_train_val_loaders
from models.make_model import build_model
from models.utils import save_checkpoint  # Ensure this is your shared save utility
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import average_precision_score


def generate_model_name(config):
    stage = "stage1"
    strategy = config.get("variant", "na")
    model = config.get("model", "clip")
    dataset = config.get("dataset", "unk")
    aspect = config.get("aspect", "unk")
    epochs = f"e{config.get('epochs', 'x')}"
    lr = f"lr{str(config.get('lr', 'x')).replace('.', '').replace('-', '')}"
    batch = f"bs{config.get('batch_size', 'x')}"
    loss = f"loss{config.get('loss', 'ce')}"
    return f"{stage}_{strategy}_{model}_{dataset}_{aspect}_{epochs}_{lr}_{batch}_{loss}"


class FinetuneTrainerStage1:
    def __init__(self, clip_model, classifier, train_loader, val_loader, config, device):
        self.clip_model = clip_model
        self.classifier = classifier.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = config["epochs"]
        self.lr = config["lr"]
        self.save_path = config["save_path"]
        self.log_path = config["log_path"]
        self.csv_path = self.log_path.replace(".txt", ".csv")
        self.config = config

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            list(self.clip_model.visual.parameters()) + list(self.classifier.parameters()), lr=self.lr
        )

        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        logging.basicConfig(
            filename=self.log_path,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filemode="w"
        )
        self.logger = logging.getLogger()
        self.logger.addHandler(logging.StreamHandler())

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch", "train_loss", "train_acc1", "train_acc5", "train_acc10",
                "val_loss", "val_acc1", "val_acc5", "val_acc10", "learning_rate", "epoch_time_sec"
            ])

    def train(self):
        self.clip_model.train()
        best_acc1 = 0.0

        # # ===== DEBUG: Check if text encoder is frozen =====
        # self.logger.info("[DEBUG] Checking if text encoder is frozen:")
        # for name, param in self.clip_model.named_parameters():
        #     if (
        #             name.startswith("transformer.") or
        #             "token_embedding" in name or
        #             "text_projection" in name
        #     ):
        #         self.logger.info(f"{'NOT FROZEN' if param.requires_grad else 'FROZEN'} {name}")
        # frozen_check = all(not p.requires_grad for n, p in self.clip_model.named_parameters()
        #                    if n.startswith("transformer.") or "token_embedding" in n or "text_projection" in n)
        # self.logger.info(f"[SUMMARY] Text encoder is {'FROZEN' if frozen_check else 'NOT FROZEN'}\n")
        # # ====================================================

        for epoch in range(1, self.epochs + 1):
            start_time = time.time()
            total_loss, total = 0.0, 0
            correct_rank1 = correct_rank5 = correct_rank10 = 0

            for batch_idx, (images, labels) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} Training")):
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                features = self.clip_model.encode_image(images)
                outputs = self.classifier(features)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, pred_rankk = outputs.topk(10, dim=1)
                total += labels.size(0)
                correct_rank1 += (pred_rankk[:, :1] == labels.unsqueeze(1)).sum().item()
                correct_rank5 += (pred_rankk[:, :5] == labels.unsqueeze(1)).sum().item()
                correct_rank10 += (pred_rankk[:, :10] == labels.unsqueeze(1)).sum().item()

                if batch_idx == 0:
                    self.logger.info(f"[DEBUG] Batch 0 - image shape: {images.shape}, labels: {labels.shape}")
                    self.logger.info(f"[DEBUG] img_features: {features.shape}, prompt: N/A, pooled: {outputs.shape}")
                    self.logger.info(f"[DEBUG] logits std: {outputs.std().item():.4f}")

            avg_loss = total_loss / len(self.train_loader)
            acc1 = 100.0 * correct_rank1 / total
            acc5 = 100.0 * correct_rank5 / total
            acc10 = 100.0 * correct_rank10 / total
            epoch_time = time.time() - start_time

            self.logger.info(f"[Epoch {epoch}] Total Loss={avg_loss:.4f}")
            self.logger.info(f"[Epoch {epoch}] Acc@1={acc1:.2f}% | Acc@5={acc5:.2f}% | Acc@10={acc10:.2f}%")

            val_metrics = self.validate()

            self.logger.info(f"Validation Results (for REID perspective)")
            self.logger.info(f"Rank-1 Accuracy     : {val_metrics['rank1']:.2f}%")
            self.logger.info(f"Rank-5 Accuracy     : {val_metrics['rank5']:.2f}%")
            self.logger.info(f"Rank-10 Accuracy    : {val_metrics['rank10']:.2f}%")
            self.logger.info(f"Mean Average Precision (mAP): {val_metrics['mean_ap']:.2f}%")

            # If classifier is present, measure classification accuracy too
            if self.classifier is not None:
                correct1 = correct5 = correct10 = 0
                total = 0
                with torch.no_grad():
                    for images, labels in self.val_loader:
                        images, labels = images.to(self.device), labels.to(self.device)
                        feats = self.clip_model.encode_image(images)
                        outputs = self.classifier(feats)
                        _, preds = outputs.topk(10, dim=1)
                        total += labels.size(0)
                        correct1 += (preds[:, :1] == labels.unsqueeze(1)).sum().item()
                        correct5 += (preds[:, :5] == labels.unsqueeze(1)).sum().item()
                        correct10 += (preds[:, :10] == labels.unsqueeze(1)).sum().item()

                cls_acc1 = 100.0 * correct1 / total
                cls_acc5 = 100.0 * correct5 / total
                cls_acc10 = 100.0 * correct10 / total

                self.logger.info(f"Validation Results (for Classification perspective)")
                self.logger.info(f"[Classifier Validation] Acc@1: {cls_acc1:.2f}% | Acc@5: {cls_acc5:.2f}% | Acc@10: {cls_acc10:.2f}%")

            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch, avg_loss, acc1, acc5, acc10,
                    val_metrics['avg_val_loss'], val_metrics['rank1'],
                    val_metrics['rank5'], val_metrics['rank10'],
                    self.lr, epoch_time
                ])

            if epoch == 1 or val_metrics['rank1'] > best_acc1:
                best_acc1 = val_metrics['rank1']
                model_name = generate_model_name(self.config) + "_BEST.pth"
                best_model_path = os.path.join(self.config['save_dir'], model_name)

                # # ===== DEBUG: Check if text encoder is frozen =====
                # self.logger.info("[DEBUG] Checking if text encoder is frozen:")
                # for name, param in self.clip_model.named_parameters():
                #     if (
                #             name.startswith("transformer.") or
                #             "token_embedding" in name or
                #             "text_projection" in name
                #     ):
                #         self.logger.info(f"{'NOT FROZEN' if param.requires_grad else 'FROZEN'} {name}")
                # frozen_check = all(not p.requires_grad for n, p in self.clip_model.named_parameters()
                #                    if n.startswith("transformer.") or "token_embedding" in n or "text_projection" in n)
                # self.logger.info(f"[SUMMARY] Text encoder is {'FROZEN' if frozen_check else 'NOT FROZEN'}\n")
                # # ====================================================


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
                self.logger.info(f"[INFO] No improvement in Rank-1 ({val_metrics['rank1']:.2f}%), skipping checkpoint.")

        model_name = generate_model_name(self.config) + "_FINAL.pth"
        final_model_path = os.path.join(self.config['save_dir'], model_name)

        # # ===== DEBUG: Check if text encoder is frozen =====
        # self.logger.info("[DEBUG] Checking if text encoder is frozen:")
        # for name, param in self.clip_model.named_parameters():
        #     if (
        #             name.startswith("transformer.") or
        #             "token_embedding" in name or
        #             "text_projection" in name
        #     ):
        #         self.logger.info(f"{'NOT FROZEN' if param.requires_grad else 'FROZEN'} {name}")
        # frozen_check = all(not p.requires_grad for n, p in self.clip_model.named_parameters()
        #                    if n.startswith("transformer.") or "token_embedding" in n or "text_projection" in n)
        # self.logger.info(f"[SUMMARY] Text encoder is {'FROZEN' if frozen_check else 'NOT FROZEN'}\n")
        # # ====================================================

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

    def validate(self):
        self.clip_model.eval()
        self.classifier.eval()

        all_feats = []
        all_labels = []
        total_val_loss = 0.0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                feats = self.clip_model.encode_image(images)
                outputs = self.classifier(feats)
                loss = self.criterion(outputs, labels)

                total_val_loss += loss.item()
                all_feats.append(feats.cpu())
                all_labels.append(labels.cpu())

        all_feats = torch.cat(all_feats, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_feats = F.normalize(all_feats, dim=1)

        self.logger.info(f"[DEBUG] Val embedding norm mean: {all_feats.norm(dim=1).mean():.4f}")

        sim_matrix = torch.matmul(all_feats, all_feats.T)
        N = len(all_labels)
        cmc = np.zeros(N)
        all_ap = []

        for i in range(N):
            query_label = all_labels[i].item()
            sim_scores = sim_matrix[i]
            sim_scores[i] = -1  # exclude self-match

            sorted_indices = torch.argsort(sim_scores, descending=True)
            matches = (all_labels[sorted_indices] == query_label).numpy().astype(int)

            # CMC
            rank_idx = np.where(matches == 1)[0]
            if len(rank_idx) == 0:
                continue
            cmc[rank_idx[0]:] += 1

            # mAP
            try:
                ap = average_precision_score(matches, sim_scores[sorted_indices].numpy())
                all_ap.append(ap)
            except ValueError:
                pass  # edge case: only one class in matches

        cmc = cmc / N
        mean_ap = np.mean(all_ap)

        return {
            'avg_val_loss': total_val_loss / len(self.val_loader),
            'rank1': 100.0 * cmc[0],
            'rank5': 100.0 * cmc[4] if len(cmc) > 4 else 0.0,
            'rank10': 100.0 * cmc[9] if len(cmc) > 9 else 0.0,
            'mean_ap': 100.0 * mean_ap
        }

def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    exp_name = generate_model_name(config)
    config["experiment"] = exp_name
    config["save_path"] = os.path.join(config["save_dir"], f"{exp_name}.pth")
    config["log_path"] = os.path.join(config["log_dir"], f"{exp_name}.log")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, num_classes = get_train_val_loaders(config)
    config["num_classes"] = num_classes

    clip_model, classifier = build_model(config, freeze_text=True)
    trainer = FinetuneTrainerStage1(clip_model, classifier, train_loader, val_loader, config, device)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    args = parser.parse_args()
    main(args.config)

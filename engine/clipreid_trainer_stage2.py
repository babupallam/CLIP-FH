import os
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime

from loss.triplet_loss import TripletLoss


def compute_topk_acc(similarity, labels, k=5):
    """
    Compute top-K accuracy for an NxN similarity matrix, where:
      - similarity[i, j] is the similarity of image i with text j
      - labels[i] is the class label for row i
      - labels[j] is the class label for column j
    We consider a match correct if labels[i] == labels[j] for at least one of
    the top k highest-similarity positions in row i.

    Returns a scalar float in [0,1].
    """
    k = min(k, similarity.size(1))
    topk_indices = similarity.topk(k, dim=1).indices  # shape: (N, k)
    # For each row i, check if any of topk_indices[i] has the same label as i.
    correct_matrix = (labels.unsqueeze(1) == labels[topk_indices])  # shape: (N, k)
    topk_correct = correct_matrix.any(dim=1).float().mean().item()
    return topk_correct


class PromptLearnerTrainerStage2b:
    def __init__(self, clip_model, prompt_learner, train_loader, config, device):
        self.clip_model = clip_model
        self.prompt_learner = prompt_learner
        self.train_loader = train_loader
        self.config = config
        self.device = device

        self.epochs = config.get("epochs", 20)
        self.lr = config.get("lr", 1e-4)
        self.batch_size = config.get("batch_size", 32)

        # === Output paths ===
        self.exp_name = config["experiment"]  # e.g., "stage2b_finetune"
        self.model = config.get("model", "vitb16")
        self.dataset = config.get("dataset", "11k")
        self.aspect = config.get("aspect", "dorsal_r")
        self.n_ctx = config.get("n_ctx", 8)
        self.freeze_text_encoder = config.get("freeze_text_encoder", True)
        self.freeze_prompt = config.get("freeze_prompt", True)

        self.loss_tri_weight = config.get("loss_tri_weight", 1.0)
        self.loss_i2t_weight = config.get("loss_i2t_weight", 1.0)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_detail = (
            f"{self.exp_name}_{self.model}_{self.dataset}_{self.aspect}"
            f"_e{self.epochs:02d}_lr{self.lr:.0e}_bs{self.batch_size}_ctx{self.n_ctx}"
            f"_freezeT{self.freeze_text_encoder}_freezeP{self.freeze_prompt}"
            f"_from2a"
            f"_lossT{self.loss_tri_weight}_I2T{self.loss_i2t_weight}"
        )

        self.save_path = os.path.join(config["save_dir"], f"{name_detail}.pth")
        self.log_path = os.path.join(config["output_dir"], f"{name_detail}.log")
        os.makedirs(config["output_dir"], exist_ok=True)
        os.makedirs(config["save_dir"], exist_ok=True)

        # === Loss Weights ===
        self.loss_weights = {
            "triplet": self.loss_tri_weight,
            "i2t": self.loss_i2t_weight,
        }

        # === Loss functions ===
        self.loss_tri = TripletLoss()
        self.loss_i2t = torch.nn.CrossEntropyLoss()

        # === Freeze text encoder and prompt learner ===
        for param in self.clip_model.transformer.parameters():
            param.requires_grad = False
        for param in self.clip_model.token_embedding.parameters():
            param.requires_grad = False
        for param in self.prompt_learner.parameters():
            param.requires_grad = False

        # === Enable training only for image encoder ===
        for param in self.clip_model.visual.parameters():
            param.requires_grad = True

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.clip_model.parameters()),
            lr=self.lr
        )

    def log(self, text):
        """Utility to print and append text to our log file."""
        print(text)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def train(self):
        self.prompt_learner.eval()
        self.clip_model.train()

        self.log("======= Stage 2b: Fine-tune Image Encoder (Debug Mode) =======")
        self.log(f"Experiment Name  : {self.exp_name}")
        self.log(f"Model & Dataset  : {self.model}, {self.dataset}, {self.aspect}")
        self.log(f"Freeze Text Enc. : {self.freeze_text_encoder}")
        self.log(f"Freeze Prompt    : {self.freeze_prompt}")
        self.log(f"Loss Weights     : Triplet={self.loss_tri_weight}, i2t={self.loss_i2t_weight}")
        self.log(f"Save Path        : {self.save_path}")
        self.log(f"LR={self.lr} | Epochs={self.epochs} | BatchSize={self.batch_size}")

        # Lists to store epoch metrics for summary
        epoch_losses = []
        epoch_tri_losses = []
        epoch_i2t_losses = []
        epoch_accs_top1 = []
        epoch_accs_top5 = []
        epoch_accs_top10 = []

        for epoch in range(self.epochs):
            epoch_start_time = time.time()

            # Tracking metrics across the epoch
            total_loss = 0.0
            total_tri_loss = 0.0
            total_i2t_loss = 0.0
            total_batches = 0

            # We'll also collect all image features & labels to compute Top-K
            all_image_feats = []
            all_text_feats = []
            all_labels = []

            # Log epoch start
            self.log(f"\n--- [Epoch {epoch + 1}/{self.epochs}] ---")

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)

                if batch_idx == 0:
                    self.log(f"[DEBUG] Batch 0 - image shape: {images.shape}, labels shape: {labels.shape}")

                # === Forward pass on image encoder (trainable) ===
                image_features = self.clip_model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)

                if (batch_idx % 50) == 0:
                    self.log(f"[DEBUG] image_features shape: {image_features.shape}")

                # === Forward pass on text side (frozen) ===
                with torch.no_grad():
                    prompts = self.prompt_learner.forward_batch(labels)
                    pos_embed = self.clip_model.positional_embedding.unsqueeze(0).expand(prompts.size(0), -1, -1)

                    x = prompts + pos_embed
                    x = x.permute(1, 0, 2)  # (context_len, B, dim)
                    x = self.clip_model.transformer(x)
                    x = x.permute(1, 0, 2)  # (B, context_len, dim)

                    text_features = self.clip_model.ln_final(x[:, 0, :])
                    text_features = F.normalize(text_features, dim=-1)

                if (batch_idx % 50) == 0:
                    self.log(f"[DEBUG] text_features shape: {text_features.shape}")

                # Store for top-k calculations
                all_image_feats.append(image_features.detach().cpu())
                all_text_feats.append(text_features.detach().cpu())
                all_labels.append(labels.detach().cpu())

                # === Build similarity matrix for i2t classification (batch-level) ===
                # shape: (B, B)
                logits = image_features @ text_features.T

                # === Compute losses ===
                loss_tri = self.loss_tri(image_features, labels)
                loss_i2t = self.loss_i2t(logits, torch.arange(logits.size(0), device=self.device))
                loss = (self.loss_weights["triplet"] * loss_tri +
                        self.loss_weights["i2t"] * loss_i2t)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_tri_loss += loss_tri.item()
                total_i2t_loss += loss_i2t.item()
                total_batches += 1

                # Show batch-level stats in the tqdm bar
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "loss_tri": f"{loss_tri.item():.4f}",
                    "loss_i2t": f"{loss_i2t.item():.4f}"
                })

                if (batch_idx % 50) == 0:
                    self.log(
                       f"[DEBUG] Step {batch_idx}: "
                        f"Loss={loss.item():.4f} | "
                        f"Tri={loss_tri.item():.4f} | "
                        f"i2t={loss_i2t.item():.4f}"
                    )

            # === End-of-batch loop, compute epoch-level metrics ===
            epoch_loss = total_loss / total_batches
            epoch_tri_loss = total_tri_loss / total_batches
            epoch_i2t_loss = total_i2t_loss / total_batches

            # === Construct NxN similarity for the entire epoch ===
            # Combine all mini-batch features
            all_image_feats = torch.cat(all_image_feats, dim=0).to(self.device)
            all_text_feats = torch.cat(all_text_feats, dim=0).to(self.device)
            all_labels = torch.cat(all_labels, dim=0).to(self.device)

            # NxN similarity
            similarity_matrix = all_image_feats @ all_text_feats.T  # (N, N)

            # === Top-1 (same as i2t diagonal approach) ===
            # We can measure how many times row i's label equals the index of i in the top-1
            # But let's do it the same way as topk
            top1_acc = compute_topk_acc(similarity_matrix, all_labels, k=1) * 100.0
            top5_acc = compute_topk_acc(similarity_matrix, all_labels, k=5) * 100.0
            top10_acc = compute_topk_acc(similarity_matrix, all_labels, k=10) * 100.0

            epoch_time = time.time() - epoch_start_time
            lr = self.optimizer.param_groups[0]["lr"]

            # === Log end-of-epoch stats ===
            self.log(
                f"[Epoch {epoch + 1}] "
                f"Total Loss={epoch_loss:.4f} | "
                f"Tri Loss={epoch_tri_loss:.4f} | "
                f"i2t Loss={epoch_i2t_loss:.4f} | "
                f"Acc@1={top1_acc:.2f}% | "
                f"Acc@5={top5_acc:.2f}% | "
                f"Acc@10={top10_acc:.2f}% | "
                f"Time={epoch_time:.2f}s | "
                f"LR={lr:.6f}"
            )

            # Store metrics for final summary
            epoch_losses.append(epoch_loss)
            epoch_tri_losses.append(epoch_tri_loss)
            epoch_i2t_losses.append(epoch_i2t_loss)
            epoch_accs_top1.append(top1_acc)
            epoch_accs_top5.append(top5_acc)
            epoch_accs_top10.append(top10_acc)

        # === Save final fine-tuned image encoder ===
        torch.save(self.clip_model.state_dict(), self.save_path)
        self.log(f"[DEBUG] Training complete. Image encoder saved to: {self.save_path}")

        # -----------
        # FINAL SUMMARY LOG
        # -----------
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_tri_loss = sum(epoch_tri_losses) / len(epoch_tri_losses)
        avg_i2t_loss = sum(epoch_i2t_losses) / len(epoch_i2t_losses)
        avg_acc_top1 = sum(epoch_accs_top1) / len(epoch_accs_top1)
        avg_acc_top5 = sum(epoch_accs_top5) / len(epoch_accs_top5)
        avg_acc_top10 = sum(epoch_accs_top10) / len(epoch_accs_top10)

        # Find the best epoch for Top-1
        best_epoch = max(range(len(epoch_accs_top1)), key=lambda i: epoch_accs_top1[i]) + 1

        self.log("\n====== FINAL TRAINING SUMMARY ======")
        self.log(f"Total epochs trained : {self.epochs}")
        self.log(f"Avg Loss            : {avg_loss:.4f}")
        self.log(f"Avg Triplet Loss    : {avg_tri_loss:.4f}")
        self.log(f"Avg i2t Loss        : {avg_i2t_loss:.4f}")
        self.log(f"Avg Acc@1           : {avg_acc_top1:.2f}%")
        self.log(f"Avg Acc@5           : {avg_acc_top5:.2f}%")
        self.log(f"Avg Acc@10          : {avg_acc_top10:.2f}%")
        self.log(f"Best Epoch (Acc@1)  : {best_epoch}")
        self.log("=====================================")

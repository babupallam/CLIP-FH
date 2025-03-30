import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from loss.contrastive_loss import clip_contrastive_loss, supcon_loss

from engine.baseline_inference import extract_features
from engine.evaluator import evaluate_rank


class PromptLearnerTrainerStage1:
    def __init__(self, clip_model, prompt_learner, train_loader, config, device):
        self.clip_model = clip_model
        self.prompt_learner = prompt_learner
        self.train_loader = train_loader
        self.config = config
        self.device = device

        self.epochs = config.get("epochs", 20)
        self.lr = config.get("lr", 1e-3)
        self.batch_size = config.get("batch_size", 32)
        self.n_ctx = config.get("n_ctx", 8)
        self.freeze_text = config.get("freeze_text_encoder", True)

        # === Generate unique experiment name based on config and timestamp ===
        exp_name = config["experiment"]
        model = config["model"]
        dataset = config["dataset"]
        aspect = config["aspect"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        name_detail = (
            f"{exp_name}_{model}_{dataset}_{aspect}"
            f"_e{self.epochs:02d}_lr{self.lr:.0e}_bs{self.batch_size}"
            f"_ctx{self.n_ctx}_freeze{str(self.freeze_text)}"
        )

        # === Output paths ===
        self.save_path = os.path.join(config["save_dir"], f"{name_detail}.pth")
        self.log_path = os.path.join(config["output_dir"], f"{name_detail}.log")
        os.makedirs(config["output_dir"], exist_ok=True)
        os.makedirs(config["save_dir"], exist_ok=True)

        # === Optimizer: only train parameters with requires_grad = True ===
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad,
                   list(self.prompt_learner.parameters()) + list(self.clip_model.parameters())),
            lr=self.lr
        )

        # === Freeze the CLIP encoders if specified (Stage 1: only train prompts) ===
        if self.freeze_text:
            # Freeze text encoder
            for param in self.clip_model.transformer.parameters():
                param.requires_grad = False
            for param in self.clip_model.token_embedding.parameters():
                param.requires_grad = False

            # Freeze image encoder (visual backbone)
            for param in self.clip_model.visual.parameters():
                param.requires_grad = False

    def log(self, text):
        """Utility to print and write log to file"""
        print(text)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def train(self):
        self.prompt_learner.train()
        self.clip_model.eval()  # image/text encoders frozen in Stage 1

        self.log(f"Experiment: {self.config['experiment']}")
        self.log(f"Save Path: {self.save_path}")
        self.log(f"Freeze Text Encoder: {self.freeze_text}")
        self.log(f"LR: {self.lr} | Epochs: {self.epochs} | BS: {self.batch_size} | N_CTX: {self.n_ctx}\n")

        for epoch in range(self.epochs):
            total_loss = 0.0
            total_batches = 0
            avg_pos_across_batches = []

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")

            for batch in pbar:
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)

                # === Step 1: Extract image features ===
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(images)

                # === Step 2: Generate prompts per label ===
                prompt_embeddings = self.prompt_learner.forward_batch(labels)  # (B, ctx_len, dim)

                # === Step 3: Add position embeddings and pass through text encoder ===
                pos_embed = self.clip_model.positional_embedding
                pos_embed = pos_embed.unsqueeze(0).expand(prompt_embeddings.size(0), -1, -1)  # (B, ctx_len, dim)

                x = prompt_embeddings + pos_embed
                x = x.permute(1, 0, 2)  # (ctx_len, B, dim) for transformer
                x = self.clip_model.transformer(x)
                x = x.permute(1, 0, 2)  # (B, ctx_len, dim)
                text_features = self.clip_model.ln_final(x[:, 0, :])  # CLS token
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # === Step 4: Compute contrastive loss ===
                loss = supcon_loss(image_features, text_features, labels)

                # === Step 5: Optional logging: average positives per sample ===
                with torch.no_grad():
                    pos_counts = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().sum(1) - 1
                    avg_pos = pos_counts.mean().item()
                    avg_pos_across_batches.append(avg_pos)

                # === Step 6: Backprop ===
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_batches += 1
                pbar.set_postfix(loss=loss.item(), avg_pos=f"{avg_pos:.2f}")

            # === Epoch logging ===
            epoch_loss = total_loss / total_batches
            avg_epoch_pos = sum(avg_pos_across_batches) / len(avg_pos_across_batches)

            self.log(f"\n[Epoch {epoch + 1}] Avg Loss: {epoch_loss:.4f}")
            self.log(f"[Epoch {epoch + 1}] Avg Positives/sample: {avg_epoch_pos:.2f}")

        # === Save learned prompt parameters ===
        torch.save(self.prompt_learner.state_dict(), self.save_path)
        self.log(f"✅ Prompt model saved to: {self.save_path}")

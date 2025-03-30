import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime

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

        # === Generate unique name ===
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

        # === Optimizer ===
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad,
                   list(self.prompt_learner.parameters()) + list(self.clip_model.parameters())),
            lr=self.lr
        )

        self.criterion = nn.CrossEntropyLoss()

        # 🔒 Freeze text encoder if required
        if self.freeze_text:
            for param in self.clip_model.transformer.parameters():
                param.requires_grad = False
            for param in self.clip_model.token_embedding.parameters():
                param.requires_grad = False

    def log(self, text):
        print(text)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def top_k_accuracy(self, logits, labels, k):
        topk_preds = logits.topk(k, dim=1).indices
        return (topk_preds == labels.unsqueeze(1)).any(dim=1).float().mean().item()

    def train(self):
        self.prompt_learner.train()
        self.clip_model.eval()

        self.log(f"Experiment: {self.config['experiment']}")
        self.log(f"Save Path: {self.save_path}")
        self.log(f"Freeze Text Encoder: {self.freeze_text}")
        self.log(f"LR: {self.lr} | Epochs: {self.epochs} | BS: {self.batch_size} | N_CTX: {self.n_ctx}")

        for epoch in range(self.epochs):
            total_loss = 0.0
            correct = 0
            top5_correct = 0
            total = 0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
            for batch in pbar:
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                # CLIP encode image
                image_features = self.clip_model.encode_image(images).detach()

                # Generate prompt embeddings
                prompt_embeddings = self.prompt_learner()

                # Add position embeddings
                pos_embed = self.clip_model.positional_embedding
                pos_embed = pos_embed.unsqueeze(0).expand(prompt_embeddings.size(0), -1, -1)
                x = prompt_embeddings + pos_embed
                x = x.permute(1, 0, 2)

                # Forward through transformer
                x = self.clip_model.transformer(x)
                x = x.permute(1, 0, 2)
                text_features = self.clip_model.ln_final(x[:, 0, :])
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Compute logits and loss
                logits = image_features @ text_features.T
                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                top5_correct += self.top_k_accuracy(logits, labels, k=5) * labels.size(0)
                total += labels.size(0)

                pbar.set_postfix(
                    loss=total_loss / (total or 1),
                    acc=correct / (total or 1),
                    top5=self.top_k_accuracy(logits, labels, k=5)
                )

            epoch_acc = correct / total
            epoch_top5 = top5_correct / total
            epoch_loss = total_loss / (total or 1)

            self.log(f"[Epoch {epoch+1}] Loss: {epoch_loss:.4f}, Acc@1: {epoch_acc:.4f}, Acc@5: {epoch_top5:.4f}")

        torch.save(self.prompt_learner.state_dict(), self.save_path)
        self.log(f"✅ Prompt model saved to: {self.save_path}")




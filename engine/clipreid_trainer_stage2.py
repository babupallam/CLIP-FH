import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from datetime import datetime
from loss.triplet_loss import TripletLoss
import time  # Add this to the top if not already imported

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
        exp_name = config["experiment"]
        model = config["model"]
        dataset = config["dataset"]
        aspect = config["aspect"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_detail = (
            f"{exp_name}_{model}_{dataset}_{aspect}"
            f"_e{self.epochs:02d}_lr{self.lr:.0e}_bs{self.batch_size}"
        )

        self.save_path = os.path.join(config["save_dir"], f"{name_detail}.pth")
        self.log_path = os.path.join(config["output_dir"], f"{name_detail}.log")
        os.makedirs(config["output_dir"], exist_ok=True)
        os.makedirs(config["save_dir"], exist_ok=True)

        # === Loss Weights ===
        self.loss_weights = {
            "triplet": config.get("loss_tri_weight", 1.0),
            "i2t": config.get("loss_i2t_weight", 1.0),
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
        print(text)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def train(self):
        self.prompt_learner.eval()
        self.clip_model.train()

        self.log(f"Running Stage 2b: Fine-tune image encoder with frozen prompts")
        self.log(f"Save Path: {self.save_path}")
        self.log(f"LR: {self.lr} | Epochs: {self.epochs} | BS: {self.batch_size}")

        for epoch in range(self.epochs):
            total_loss = 0.0
            total_batches = 0
            correct = 0
            total = 0
            start_time = time.time()

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")

            for batch in pbar:
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)

                # === Image features (trainable encoder) ===
                image_features = self.clip_model.encode_image(images)
                image_features = F.normalize(image_features, dim=-1)

                # === Text features (frozen prompts) ===
                with torch.no_grad():
                    prompts = self.prompt_learner.forward_batch(labels)
                    pos_embed = self.clip_model.positional_embedding
                    pos_embed = pos_embed.unsqueeze(0).expand(prompts.size(0), -1, -1)
                    x = prompts + pos_embed
                    x = x.permute(1, 0, 2)
                    x = self.clip_model.transformer(x)
                    x = x.permute(1, 0, 2)
                    text_features = self.clip_model.ln_final(x[:, 0, :])
                    text_features = F.normalize(text_features, dim=-1)

                # === Similarity matrix ===
                logits = image_features @ text_features.T  # shape: (B, B)

                # === Compute losses ===
                loss_tri = self.loss_tri(image_features, labels)
                loss_i2t = self.loss_i2t(logits, torch.arange(logits.size(0)).to(self.device))

                # === Combine ===
                loss = (self.loss_weights["triplet"] * loss_tri +
                        self.loss_weights["i2t"] * loss_i2t)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                total_batches += 1

                preds = logits.argmax(dim=1)
                correct += (preds == torch.arange(logits.size(0)).to(self.device)).sum().item()
                total += labels.size(0)

                pbar.set_postfix(loss=loss.item())

            # === End-of-epoch logging ===
            epoch_loss = total_loss / total_batches
            epoch_acc = 100.0 * correct / total if total > 0 else 0.0
            epoch_time = time.time() - start_time
            lr = self.optimizer.param_groups[0]["lr"]

            self.log(
                f"- Epoch {epoch + 1}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%, Time={epoch_time:.2f}s, LR={lr:.4f}")

        # Save final fine-tuned image encoder
        torch.save(self.clip_model.state_dict(), self.save_path)
        self.log(f"✅ Image encoder saved to: {self.save_path}")

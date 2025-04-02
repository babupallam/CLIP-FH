import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from loss.make_loss import build_loss

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

        # loss function accessing
        self.loss_fn = build_loss(
            loss_list=config.get("loss_list", ["supcon"]),
            num_classes=config["num_classes"],
            feat_dim=clip_model.ln_final.weight.shape[0]
        )

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

        print("🔍 Prompt Learner Parameters:")
        for name, param in self.prompt_learner.named_parameters():
            print(f" - {name}: requires_grad = {param.requires_grad}")

        # === Freeze the CLIP encoders if specified (Stage 1: only train prompts) ===
        if self.freeze_text:
            for param in self.clip_model.transformer.parameters():
                param.requires_grad = False
            for param in self.clip_model.token_embedding.parameters():
                param.requires_grad = False
            for param in self.clip_model.visual.parameters():
                param.requires_grad = False

            ## DO NOT freeze prompt learner in this

        # to test again
        print([p.requires_grad for p in self.prompt_learner.parameters()])  # should all be True

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
            start_time = time.time()

            total_loss = 0.0
            total_batches = 0
            avg_pos_across_batches = []
            row_acc_list, col_acc_list, grad_norm_list, prompt_norm_list = [], [], [], []

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}")

            for batch in pbar:
                images, labels = batch
                images, labels = images.to(self.device), labels.to(self.device)

                # === Step 1: Extract image features ===
                image_features = self.clip_model.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # === Step 2: Generate prompts per label ===
                prompt_embeddings = self.prompt_learner.forward_batch(labels)

                # === Step 3: Pass prompts through transformer ===
                pos_embed = self.clip_model.positional_embedding
                pos_embed = pos_embed.unsqueeze(0).expand(prompt_embeddings.size(0), -1, -1)
                x = prompt_embeddings + pos_embed
                x = x.permute(1, 0, 2)
                x = self.clip_model.transformer(x)
                x = x.permute(1, 0, 2)
                text_features = self.clip_model.ln_final(x[:, 0, :])
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # === Step 4:  loss fucntion ===
                loss_i2t = self.loss_fn(features=image_features, text_features=text_features, targets=labels,
                                        mode="contrastive")
                loss_t2i = self.loss_fn(features=text_features, text_features=image_features, targets=labels,
                                        mode="contrastive")
                contrastive_loss = loss_i2t + loss_t2i
                prompt_reg = (self.prompt_learner.ctx ** 2).mean()
                loss = contrastive_loss + 0.001 * prompt_reg

                # === Step 5: Logging metrics ===
                with torch.no_grad():
                    pos_counts = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().sum(1) - 1
                    avg_pos = pos_counts.mean().item()
                    avg_pos_across_batches.append(avg_pos)

                    # Similarity matrix
                    sim = image_features @ text_features.T
                    row_acc = (sim.argmax(1) == torch.arange(sim.size(0), device=self.device)).float().mean().item()
                    col_acc = (sim.argmax(0) == torch.arange(sim.size(0), device=self.device)).float().mean().item()
                    row_acc_list.append(row_acc)
                    col_acc_list.append(col_acc)

                    # Prompt norm
                    prompt_norm = self.prompt_learner.ctx.norm(dim=1).mean().item()
                    prompt_norm_list.append(prompt_norm)

                self.optimizer.zero_grad()
                loss = (self.prompt_learner.ctx ** 2).mean()
                loss.backward()
                """
                for name, param in self.prompt_learner.named_parameters():
                    if param.grad is None:
                        print(f"🚫 {name} has no grad")
                    else:
                        print(f"✅ {name} grad norm: {param.grad.norm().item():.6f}")
                """
                # Prompt gradient norm
                if self.prompt_learner.ctx.grad is not None:
                    grad_norm = self.prompt_learner.ctx.grad.norm().item()
                    grad_norm_list.append(grad_norm)

                self.optimizer.step()

                total_loss += loss.item()
                total_batches += 1
                pbar.set_postfix(loss=loss.item(), avg_pos=f"{avg_pos:.2f}", row_acc=f"{row_acc:.2f}")

            # === End of Epoch Logging ===
            epoch_loss = total_loss / total_batches
            avg_epoch_pos = sum(avg_pos_across_batches) / len(avg_pos_across_batches)
            avg_row_acc = sum(row_acc_list) / len(row_acc_list)
            avg_col_acc = sum(col_acc_list) / len(col_acc_list)
            avg_grad = sum(grad_norm_list) / len(grad_norm_list) if grad_norm_list else 0
            avg_prompt_norm = sum(prompt_norm_list) / len(prompt_norm_list)
            prompt_std = torch.std(self.prompt_learner.ctx).item()  # prompt variability
            epoch_time = time.time() - start_time

            # === Optional: Top-5 Img→Text accuracy ===
            with torch.no_grad():
                sim = image_features @ text_features.T
                top5_row_acc = (
                    (labels.unsqueeze(1) == labels[sim.topk(5, dim=1).indices]).any(dim=1).float().mean().item()
                )

            # === Unified Single-Line Logging ===
            self.log(
                f"[Epoch {epoch + 1:02d}] "
                f"Loss: {epoch_loss:.4f} | "
                f"Pos/Sample: {avg_epoch_pos:.2f} | "
                f"Img→Text@1: {avg_row_acc:.4f} | "
                f"Img→Text@5: {top5_row_acc:.4f} | "
                f"Text→Img@1: {avg_col_acc:.4f} | "
                f"PromptNorm: {avg_prompt_norm:.4f} | "
                f"PromptVar: {prompt_std:.4f} | "
                f"PromptGrad: {avg_grad:.4f} | "
                f"Time: {epoch_time:.2f}s"
            )

            # === Early stopping based on prompt norm ===
            #early_stop_threshold = self.config.get("early_stop_prompt_norm", 1.2e-3)
            #if avg_prompt_norm < early_stop_threshold:
            #    self.log(f"🛑 Early stopping: prompt norm {avg_prompt_norm:.6f} < threshold {early_stop_threshold}")
            #    break


        # === Save learned prompt parameters ===
        torch.save(self.prompt_learner.state_dict(), self.save_path)
        self.log(f"✅ Prompt model saved to: {self.save_path}")

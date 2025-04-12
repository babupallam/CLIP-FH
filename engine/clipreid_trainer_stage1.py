import os
import time
import torch
import torch.optim as optim
from tqdm import tqdm
from datetime import datetime
from utils.loss.make_loss import build_loss


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

        print(" Prompt Learner Parameters:")
        for name, param in self.prompt_learner.named_parameters():
            print(f" - {name}: requires_grad = {param.requires_grad}")

        # === Freeze the CLIP encoders if specified (Stage 1: only train prompts) ===
        if self.freeze_text:
            print("Freezing everything except prompt learner....")
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

        # === OFFLINE FEATURE CACHING ===
        image_features_all = []
        labels_all = []

        print("Extracting and caching image features from training set...")
        with torch.no_grad():
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                feats = self.clip_model.encode_image(images)
                feats = feats / feats.norm(dim=-1, keepdim=True)

                image_features_all.append(feats.cpu())
                labels_all.append(labels.cpu())

        image_features_all = torch.cat(image_features_all, dim=0).to(self.device)
        labels_all = torch.cat(labels_all, dim=0).to(self.device)
        num_samples = image_features_all.shape[0]

        for epoch in range(self.epochs):

            start_time = time.time()

            total_loss = 0.0
            avg_pos_across_batches = []
            row_acc_list, col_acc_list, grad_norm_list, prompt_norm_list = [], [], [], []

            indices = torch.randperm(num_samples)

            pbar = tqdm(range(0, num_samples, self.batch_size), desc=f"Epoch {epoch + 1}/{self.epochs}")
            for i in pbar:
                batch_idx = indices[i:i + self.batch_size]
                img_feats = image_features_all[batch_idx]
                labels = labels_all[batch_idx]

                # === Step 2: Generate prompts per label ===
                prompt_embeddings = self.prompt_learner.forward_batch(labels)

                if epoch == 0:
                    print("ctx grad_fn:", self.prompt_learner.ctx.grad_fn)
                    print("prompt_embedded requires_grad:", prompt_embeddings.requires_grad)

                # === Step 3: Prepare positional embeddings ===
                pos_embed = self.clip_model.positional_embedding


                # === Step 4: Build text features ===
                try:
                    x = prompt_embeddings + pos_embed.unsqueeze(0).expand(prompt_embeddings.size(0), -1, -1)
                    x = x.permute(1, 0, 2)  # (context_len, B, dim)
                    x = self.clip_model.transformer(x)
                    x = x.permute(1, 0, 2)  # (B, context_len, dim)
                    x = self.clip_model.ln_final(x[:, 0, :])  # [CLS] token

                    # CRITICAL LINE:
                    x = x @ self.clip_model.text_projection
                    #print("x: ",self.clip_model.text_projection.shape)

                    text_feats = x / x.norm(dim=-1, keepdim=True)
                    #print("text_feats after projection: ", x.shape, x[0, :5])


                except Exception as e:
                    print("Error during prompt→text embedding block!", e)
                    print(f"prompt_embeddings shape: {prompt_embeddings.shape}")
                    print(f"pos_embed shape: {self.clip_model.positional_embedding.shape}")
                    raise e  # re-raise to see full error

                assert img_feats is not None, "img_feats is None!"
                assert text_feats is not None, "text_feats is None!"
                assert img_feats.shape == text_feats.shape, f"Shape mismatch: {img_feats.shape} vs {text_feats.shape}"

                #print(f"text_feats: {text_feats.shape}, img_feats: {img_feats.shape}, labels: {labels.shape}")

                # === Step 4: Contrastive Loss ===
                loss_i2t = self.loss_fn(features=img_feats, text_features=text_feats, targets=labels,
                                        mode="contrastive")
                loss_t2i = self.loss_fn(features=text_feats, text_features=img_feats, targets=labels,
                                        mode="contrastive")

                contrastive_loss = loss_i2t + loss_t2i

                prompt_reg = (self.prompt_learner.ctx ** 2).mean()
                loss = contrastive_loss + 0.001 * prompt_reg


                # === Step 5: Logging metrics ===
                with torch.no_grad():
                    pos_counts = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().sum(1) - 1
                    avg_pos = pos_counts.mean().item()
                    avg_pos_across_batches.append(avg_pos)

                    sim = img_feats @ text_feats.T
                    row_acc = (sim.argmax(1) == torch.arange(sim.size(0), device=self.device)).float().mean().item()
                    col_acc = (sim.argmax(0) == torch.arange(sim.size(0), device=self.device)).float().mean().item()
                    row_acc_list.append(row_acc)
                    col_acc_list.append(col_acc)

                    prompt_norm = self.prompt_learner.ctx.norm(dim=1).mean().item()
                    prompt_norm_list.append(prompt_norm)

                self.optimizer.zero_grad()
                self.prompt_learner.ctx.retain_grad()  # ensure ctx grad is retained
                loss.backward()

                if self.prompt_learner.ctx.grad is None:
                    print(" Warning: No gradient received by prompt learner. Check data or loss.")
                else:
                    grad_norm = self.prompt_learner.ctx.grad.norm().item()
                    grad_norm_list.append(grad_norm)
                    if epoch == 0 and i == 0:
                        print(" ctx.grad example:", self.prompt_learner.ctx.grad[0][:5])

                self.optimizer.step()
                total_loss += loss.item()

                pbar.set_postfix(loss=loss.item(), avg_pos=f"{avg_pos:.2f}", row_acc=f"{row_acc:.2f}")
            #print(grad_norm_list)

            # === End of Epoch Logging ===
            epoch_loss = total_loss / len(pbar)
            avg_epoch_pos = sum(avg_pos_across_batches) / len(avg_pos_across_batches)
            avg_row_acc = sum(row_acc_list) / len(row_acc_list)
            avg_col_acc = sum(col_acc_list) / len(col_acc_list)
            avg_grad = sum(grad_norm_list) / len(grad_norm_list) if grad_norm_list else 0
            avg_prompt_norm = sum(prompt_norm_list) / len(prompt_norm_list)
            prompt_std = torch.std(self.prompt_learner.ctx).item()
            epoch_time = time.time() - start_time

            with torch.no_grad():
                sim = img_feats @ text_feats.T
                k = min(5, sim.size(1))
                top5_row_acc = compute_topk_acc(sim, labels, k)

            self.log(
                f"[Epoch {epoch + 1:02d}] "
                f"Loss: {epoch_loss:.4f} | "
                f"Pos/Sample: {avg_epoch_pos:.2f} | "
                f"Img→Text@1: {avg_row_acc:.4f} | "
                f"Img→Text@5: {top5_row_acc:.4f} | "
                f"Text→Img@1: {avg_col_acc:.4f} | "
                f"PromptNorm: {avg_prompt_norm:.4f} | "
                f"PromptVar: {prompt_std:.4f} | "
                f"PromptGrad: {avg_grad:.7f} | "
                f"Time: {epoch_time:.2f}s"
            )

        # === Save prompt model only ===
        torch.save(self.prompt_learner.state_dict(), self.save_path)
        self.log(f"Prompt model saved to: {self.save_path}")




def compute_topk_acc(similarity, labels, k=5):
    k = min(k, similarity.size(1))
    topk_indices = similarity.topk(k, dim=1).indices
    return (
        (labels.unsqueeze(1) == labels[topk_indices])
        .any(dim=1)
        .float()
        .mean()
        .item()
    )

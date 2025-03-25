import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from loss.make_loss import build_loss

class ClipReIDImageEncoderTrainer:
    def __init__(self, clip_model, prompt_learner, train_loader, config, device):
        self.clip_model = clip_model
        self.prompt_learner = prompt_learner
        self.train_loader = train_loader
        self.device = device

        self.epochs = config["epochs"]
        self.lr = config["lr"]
        self.log_path = config["log_path"]
        self.save_path = config["save_path"]
        self.num_classes = config["num_classes"]
        self.feat_dim = clip_model.visual.output_dim

        # 🔒 Freeze text encoder & prompt learner
        for p in self.clip_model.transformer.parameters():
            p.requires_grad = False
        for p in self.prompt_learner.parameters():
            p.requires_grad = False

        # ✅ Enable image encoder gradients
        for p in self.clip_model.visual.parameters():
            p.requires_grad = True

        # ✅ Build loss
        self.criterion = build_loss(
            loss_list=config["loss"],
            num_classes=self.num_classes,
            feat_dim=self.feat_dim
        )

        # ✅ Optimizer: only update image encoder
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.clip_model.parameters()),
            lr=self.lr
        )

    def train(self):
        self.clip_model.train()
        self.prompt_learner.eval()
        log = []

        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            total = 0
            correct = 0

            # Freeze prompt to generate class text embeddings
            with torch.no_grad():
                prompts = self.prompt_learner().to(self.device)
                text_features = self.clip_model.encode_text_from_embedding(prompts)
                text_features = F.normalize(text_features, dim=1)

            for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                # 🔁 Encode image
                image_features = self.clip_model.encode_image(images)
                image_features = F.normalize(image_features, dim=1)

                # Classification logits
                logits = image_features @ text_features.T  # (B, C)

                # Compute combined loss
                loss = self.criterion(outputs=logits, targets=labels, features=image_features)

                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            acc = 100. * correct / total
            avg_loss = total_loss / len(self.train_loader)
            line = f"Epoch {epoch}: Loss={avg_loss:.4f}, Acc={acc:.2f}%"
            print(line)
            log.append(line)

        # ✅ Save fine-tuned model
        torch.save(self.clip_model.visual.state_dict(), self.save_path)
        with open(self.log_path, "w") as f:
            f.write("\n".join(log))
        print(f"\n✅ Fine-tuned image encoder saved to: {self.save_path}")

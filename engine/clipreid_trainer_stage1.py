import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F

class ClipReIDPromptTrainer:
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

        # Freeze CLIP (text and image encoders)
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # Only train prompt tokens
        self.optimizer = optim.Adam(prompt_learner.parameters(), lr=self.lr)

    def train(self):
        self.clip_model.eval()  # Keep CLIP frozen
        self.prompt_learner.train()
        log = []

        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            total = 0
            correct = 0

            for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                # 🔁 Recompute text features for this batch
                prompts = self.prompt_learner().to(self.device)  # (num_classes, ctx_len, dim)
                text_features = self.clip_model.encode_text_from_embedding(prompts)
                text_features = F.normalize(text_features, dim=1)

                # 🔁 Encode image features
                image_features = self.clip_model.encode_image(images)
                image_features = F.normalize(image_features, dim=1)

                # Cosine similarity → logits → CE loss
                logits = image_features @ text_features.T  # (B, C)
                loss = F.cross_entropy(logits, labels)

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

        # ✅ Save prompt learner weights
        torch.save(self.prompt_learner.state_dict(), self.save_path)
        with open(self.log_path, "w") as f:
            f.write("\n".join(log))
        print(f"\n✅ Prompt learner saved to: {self.save_path}")

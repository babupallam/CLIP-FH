import torch
import torch.optim as optim
from tqdm import tqdm
from loss.make_loss import build_loss

class FinetuneTrainerStage2:
    def __init__(self, clip_model, classifier, train_loader, config, device):
        self.clip_model = clip_model
        self.classifier = classifier.to(device)
        self.train_loader = train_loader
        self.device = device

        self.epochs = config["epochs"]
        self.lr = config["lr"]
        self.save_path = config["save_path"]
        self.log_path = config["log_path"]
        self.loss_list = config["loss"]

        image_feat_dim = clip_model.visual.output_dim
        num_classes = config["num_classes"]
        self.loss_fn = build_loss(self.loss_list, num_classes, image_feat_dim)

        self.optimizer = optim.Adam(
            list(self.clip_model.visual.parameters()) + list(self.classifier.parameters()),
            lr=self.lr
        )

    def train(self):
        self.clip_model.train()
        log = []

        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            total = 0
            correct = 0

            for images, labels in tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs}"):
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                with torch.no_grad():
                    features = self.clip_model.encode_image(images)

                outputs = self.classifier(features)
                loss = self.loss_fn(outputs, labels, features)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, preds = outputs.max(1)
                total += labels.size(0)
                correct += preds.eq(labels).sum().item()

            acc = 100.0 * correct / total
            avg_loss = total_loss / len(self.train_loader)
            log_line = f"Epoch {epoch}: Loss={avg_loss:.4f}, Acc={acc:.2f}%"
            print(log_line)
            log.append(log_line)

        # Save model + logs
        torch.save(self.clip_model.state_dict(), self.save_path)
        with open(self.log_path, "w") as f:
            f.write("\n".join(log))
        print(f"\n✅ Model saved to: {self.save_path}")
        print(f"📝 Training log saved to: {self.log_path}")

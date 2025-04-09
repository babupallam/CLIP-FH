import torch  # This is like the main tool for building and training our "smart" program.
import torch.nn as nn  # This helps us build the different parts of our "smart" program.
import torch.optim as optim  # This helps our "smart" program learn from its mistakes.
from tqdm import tqdm  # This makes progress bars so we can see how our training is going.
import os  # This helps us work with files and folders on our computer.
import logging  # This helps us keep track of what's happening during training.
import time  # This helps us measure how long things take.
import csv  # This helps us save our training results in a spreadsheet-like format.

class FinetuneTrainerStage1:
    def __init__(self, clip_model, classifier, train_loader, val_loader, config, device):
        """
        This is like setting up our "smart" program with all the things it needs.
        """
        self.clip_model = clip_model  # This is the part of our program that understands pictures.
        self.classifier = classifier.to(device)  # This is the part that makes guesses about the pictures.
        self.train_loader = train_loader  # This gives our program the pictures and answers to learn from.
        self.val_loader = val_loader  # This gives our program pictures to test how well it's learned.
        self.device = device  # This says where to do the learning (like a fast or slow computer).

        self.epochs = config["epochs"]  # This is how many times our program will look at all the training pictures.
        self.lr = config["lr"]  # This is how quickly our program learns.
        self.save_path = config["save_path"]  # This is where we save our trained program.
        self.log_path = config["log_path"]  # This is where we save a log of what happened during training.
        self.csv_path = self.log_path.replace(".txt", ".csv")  # This is where we save our training results in a spreadsheet.

        self.criterion = nn.CrossEntropyLoss()  # This is how we measure how wrong our program's guesses are.
        self.optimizer = optim.Adam(  # This is how we help our program learn from its mistakes.
            list(self.clip_model.visual.parameters()) + list(self.classifier.parameters()),
            lr=self.lr
        )

        # Setting up the log file
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        logging.basicConfig(
            filename=self.log_path,
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            filemode="w"
        )
        self.logger = logging.getLogger()
        self.logger.addHandler(logging.StreamHandler())

        # Setting up the spreadsheet file
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc1", "train_acc5", "train_acc10", "val_loss", "val_acc1", "val_acc5", "val_acc10", "learning_rate", "epoch_time_sec"])

        self.config = config  # Saving the settings so we can use them later.

    def train(self):
        """
        This is where we teach our program to recognize pictures.
        """
        self.clip_model.train()  # Telling the program it's time to learn.
        best_acc1 = 0.0  # Keeping track of the best score our program got on the test.
        best_epoch = 0  # Keeping track of when our program got the best score.

        for epoch in range(1, self.epochs + 1):  # Going through all the training many times.
            start_time = time.time()  # Remembering when we started this round of training.
            total_loss = 0.0  # Keeping track of how wrong our program's guesses were.
            total = 0  # Keeping track of how many pictures our program has seen.
            correct_top1 = correct_top5 = correct_top10 = 0  # Keeping track of how many guesses were right.

            for batch_idx, (images, labels) in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.epochs} Training")):
                images, labels = images.to(self.device), labels.to(self.device)  # Giving the program a batch of pictures.

                self.optimizer.zero_grad()  # Resetting the program's memory of past mistakes.
                features = self.clip_model.encode_image(images)  # Getting the program to understand the pictures.
                outputs = self.classifier(features)  # Getting the program to make guesses.
                loss = self.criterion(outputs, labels)  # Checking how wrong the guesses were.
                loss.backward()  # Telling the program how to adjust itself to make better guesses.
                self.optimizer.step()  # Letting the program adjust itself.

                total_loss += loss.item()  # Adding up how wrong the guesses were.
                _, pred_topk = outputs.topk(10, dim=1)  # Getting the top 10 guesses.
                total += labels.size(0)  # Counting how many pictures we've seen.
                correct_top1 += (pred_topk[:, :1] == labels.unsqueeze(1)).sum().item()  # Counting how many top guesses were right.
                correct_top5 += (pred_topk[:, :5] == labels.unsqueeze(1)).sum().item()  # Counting how many top 5 guesses were right.
                correct_top10 += (pred_topk[:, :10] == labels.unsqueeze(1)).sum().item() # Counting how many top 10 guesses were right.

                if batch_idx == 0:  # If this is the first batch, we'll log some info.
                    self.logger.info(f"[DEBUG] Batch 0 - image shape: {images.shape}, labels: {labels.shape}")
                    self.logger.info(f"[DEBUG] img_features: {features.shape}, prompt: N/A, pooled: {outputs.shape}")

            avg_loss = total_loss / len(self.train_loader)  # Getting the average wrongness.
            acc1 = 100.0 * correct_top1 / total  # Getting the percentage of top guesses that were right.
            acc5 = 100.0 * correct_top5 / total  # Getting the percentage of top 5 guesses that were right.
            acc10 = 100.0 * correct_top10 / total # Getting the percentage of top 10 guesses that were right.
            epoch_time = time.time() - start_time  # Seeing how long this round took.

            val_metrics = self.validate()  # Testing how well the program has learned.

            # Logging the results of this round.
            self.logger.info(f"[Epoch {epoch}] Total Loss={avg_loss:.4f}")
            self.logger.info(f"[Epoch {epoch}] Acc@1={acc1:.2f}% | Acc@5={acc5:.2f}% | Acc@10={acc10:.2f}%")

            self.logger.info("\n" + "=" * 20 + f" Epoch {epoch} Validation " + "=" * 20)
            self.logger.info(f"Avg Validation Loss : {val_metrics['avg_val_loss']:.4f}")
            self.logger.info(f"Validation Top-1    : {val_metrics['top1_accuracy']:.2f}%")
            self.logger.info(f"Validation Top-5    : {val_metrics['top5_accuracy']:.2f}%")
            self.logger.info(f"Validation Top-10   : {val_metrics['top10_accuracy']:.2f}%")
            self.logger.info("=" * 60 + "\n")

            # Saving the results to the spreadsheet.
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch, avg_loss, acc1, acc5, acc10, val_metrics['avg_val_loss'], val_metrics['top1_accuracy'], val_metrics['top5_accuracy'], val_metrics['top10_accuracy'], self.lr, epoch_time])

            # If the program did better on the test, we'll save it.
            if val_metrics['top1_accuracy'] > best_acc1:
                best_acc1 = val_metrics['top1_accuracy']
                best_epoch = epoch

                # Making a name for the saved program.
                model_name = (
                    f"stage1_frozen_text_{self.config.get('clip_model', 'vitb16').replace('/', '')}_"
                    f"{self.config.get('dataset', 'unk')}_{self.config.get('aspect', 'unk')}_e{self.epochs}_"
                    f"lr{str(self.lr).replace('.', 'p')}_bs{self.config.get('batch_size', 'unk')}_"
                    f"{self.config.get('loss', 'crossentropy')}_BEST.pth"
                )

                best_model_path = os.path.join(self.config['save_dir'], model_name)

                # Saving the program's "memory".
                torch.save({
                    'epoch': epoch,
                    'clip_visual_state_dict': self.clip_model.visual.state_dict(),
                    'classifier_state_dict': self.classifier.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': avg_loss,
                    'config': self.config,
                    'top1_accuracy': val_metrics['top1_accuracy']
                }, best_model_path)

                self.logger.info(f"Saving best model at epoch {epoch} (Acc@1={val_metrics['top1_accuracy']:.2f}%) -> {best_model_path}")

        # Saving the program's "memory" one last time.
        model_name = (
            f"stage1_frozen_text_{self.config.get('clip_model', 'vitb16').replace('/', '')}_"
            f"{self.config.get('dataset', 'unk')}_{self.config.get('aspect', 'unk')}_e{self.epochs}_"
            f"lr{str(self.lr).replace('.', 'p')}_bs{self.config.get('batch_size', 'unk')}_"
            f"{self.config.get('loss', 'crossentropy')}_FINAL.pth"
        )
        final_model_path = os.path.join(self.config['save_dir'], model_name)
        torch.save(self.clip_model.state_dict(), final_model_path)
        self.logger.info(f"Model saved to: {final_model_path}")

    def validate(self):
        """
        This is where we test how well our program has learned.
        """
        self.clip_model.eval()  # Telling the program it's time to test.
        self.classifier.eval()

        total_val_loss = 0.0  # Keeping track of how wrong the program's guesses were on the test.
        total = 0  # Keeping track of how many test pictures we've seen.
        correct_top1 = correct_top5 = correct_top10 = 0  # Keeping track of how many guesses were right.

        with torch.no_grad():  # Telling the program not to try to learn during the test.
            for images, labels in self.val_loader:  # Going through all the test pictures.
                images, labels = images.to(self.device), labels.to(self.device)  # Giving the program a test picture.

                features = self.clip_model.encode_image(images)  # Getting the program to understand the picture.
                outputs = self.classifier(features)  # Getting the program to make a guess.
                loss = self.criterion(outputs, labels)  # Checking how wrong the guess was.

                total_val_loss += loss.item()  # Adding up how wrong the guesses were.
                _, pred_topk = outputs.topk(10, dim=1)  # Getting the top 10 guesses.
                total += labels.size(0)  # Counting how many test pictures we've seen.
                correct_top1 += (pred_topk[:, :1] == labels.unsqueeze(1)).sum().item()  # Counting how many top guesses were right.
                correct_top5 += (pred_topk[:, :5] == labels.unsqueeze(1)).sum().item()  # Counting how many top 5 guesses were right.
                correct_top10 += (pred_topk[:, :10] == labels.unsqueeze(1)).sum().item()  # Counting how many top 10 guesses were right.

        avg_val_loss = total_val_loss / len(self.val_loader)  # Getting the average wrongness on the test.
        acc1 = 100.0 * correct_top1 / total  # Getting the percentage of top guesses that were right on the test.
        acc5 = 100.0 * correct_top5 / total  # Getting the percentage of top 5 guesses that were right on the test.
        acc10 = 100.0 * correct_top10 / total  # Getting the percentage of top 10 guesses that were right on the test.

        # Returning the results of the test.
        return {
            'avg_val_loss': avg_val_loss,
            'top1_accuracy': acc1,
            'top5_accuracy': acc5,
            'top10_accuracy': acc10
        }
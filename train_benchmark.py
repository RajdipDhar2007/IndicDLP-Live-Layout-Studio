import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

REPO_DIR = Path(__file__).resolve().parent
CHECKPOINTS_DIR = REPO_DIR / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

if str(REPO_DIR) not in sys.path:
    sys.path.append(str(REPO_DIR))

from ad_dataset import HistoricalAdDataset
from models_benchmark import LeNet5, build_resnet18

def parse_args():
    parser = argparse.ArgumentParser(description="Train Ad vs Non-Ad Classifier using Udacity PyTorch Template")
    parser.add_argument("--data-dir", type=str, default=str(REPO_DIR / "Ad_Dataset"), help="Dataset directory")
    parser.add_argument("--model", type=str, default="resnet18", choices=["lenet", "resnet18"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=32, help="Input batch size for training")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    return parser.parse_args()

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, device, num_epochs=8, checkpoint_path=None):
    since = time.time()
    best_val_f1 = 0.0

    print(f"\nTraining on: {device}")
    print("-" * 65)

    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        print("-" * 15)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            all_preds = []
            all_targets = []
            all_probs = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    probs = torch.softmax(outputs, dim=1)[:, 1]
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_probs.extend(probs.detach().cpu().tolist())
                all_preds.extend(preds.detach().cpu().tolist())
                all_targets.extend(labels.detach().cpu().tolist())

            if phase == "train" and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = accuracy_score(all_targets, all_preds)
            p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="binary", pos_label=1, zero_division=0)

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc*100:.2f}% | Ad-P: {p:.3f} Ad-R: {r:.3f} Ad-F1: {f1:.3f}")

            if phase == "val" and f1 > best_val_f1:
                print(f"Validation Ad-F1 increased ({best_val_f1:.4f} --> {f1:.4f}). Saving model...")
                best_val_f1 = f1
                if checkpoint_path:
                    torch.save(model.state_dict(), checkpoint_path)

        print()

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best Val Ad-F1: {best_val_f1:.4f}")
    return model

def evaluate_test_set(model, test_loader, dataset_size, criterion, device):
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)

            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.cpu().tolist())

    avg_test_loss = test_loss / dataset_size
    acc = accuracy_score(all_targets, all_preds)
    p, r, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="binary", pos_label=1, zero_division=0)
    macro_f1 = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)[2]

    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 0.0

    cm = confusion_matrix(all_targets, all_preds)

    print("=" * 65)
    print("TEST SET EVALUATION RESULTS:")
    print("=" * 65)
    print(f"Test Loss:        {avg_test_loss:.4f}")
    print(f"Test Accuracy:    {acc*100:.2f}%")
    print(f"Ad Precision:     {p*100:.2f}%")
    print(f"Ad Recall:        {r*100:.2f}%")
    print(f"Ad F1-Score:      {f1*100:.2f}%")
    print(f"Macro F1:         {macro_f1*100:.2f}%")
    print(f"ROC-AUC:          {auc:.4f}")
    print("\nConfusion Matrix [TN, FP / FN, TP]:")
    print(cm)
    print("=" * 65)

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    train_on_gpu = torch.cuda.is_available()
    device = torch.device("cuda:0" if train_on_gpu else "cpu")
    if train_on_gpu:
        print(f"CUDA is available! Training on {torch.cuda.get_device_name(0)} ...")
    else:
        print("CUDA is not available. Training on CPU ...")

    image_datasets = {
        "train": HistoricalAdDataset(data_dir, split="train"),
        "val": HistoricalAdDataset(data_dir, split="val"),
        "test": HistoricalAdDataset(data_dir, split="test")
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val", "test"]}

    dataloaders = {
        "train": DataLoader(image_datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=0),
        "val": DataLoader(image_datasets["val"], batch_size=args.batch_size, shuffle=False, num_workers=0),
        "test": DataLoader(image_datasets["test"], batch_size=args.batch_size, shuffle=False, num_workers=0)
    }

    print(f"Dataset summary: Train={dataset_sizes['train']}, Val={dataset_sizes['val']}, Test={dataset_sizes['test']}")

    n_ad = sum(1 for _, l in image_datasets["train"].samples if l == 1)
    n_non_ad = dataset_sizes["train"] - n_ad
    w_non_ad = dataset_sizes["train"] / (2.0 * n_non_ad)
    w_ad = dataset_sizes["train"] / (2.0 * n_ad)
    weights = torch.tensor([w_non_ad, w_ad], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    checkpoint_file = CHECKPOINTS_DIR / f"best_{args.model}.pt"

    if args.model == "lenet":
        model = LeNet5(num_classes=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 2)
    else:
        model = build_resnet18(num_classes=2, pretrained=True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

    train_model(
        model=model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        checkpoint_path=checkpoint_file
    )

    model.load_state_dict(torch.load(checkpoint_file, map_location=device, weights_only=True))
    evaluate_test_set(model, dataloaders["test"], dataset_sizes["test"], criterion, device)

if __name__ == "__main__":
    main()

import os
import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from transformers import DataCollatorWithPadding

from config import *
from data import train_dataset, val_dataset

from model import MultiHeadSigmoid
from trainer import train_epoch, eval_epoch

# from utils import calculate_alpha
from record import upload_blob
from preprocess import tokenizer


def main(args):
    print(f"START WITH ARGS: {args}")

    train_data = train_dataset
    val_data = val_dataset
    running_epochs = args.epochs
    if args.run_mode == "sanity_check":
        train_data = Subset(train_dataset, range(SUBSET_SIZE))
        val_data = Subset(val_dataset, range(SUBSET_SIZE))
        running_epochs = 5
        print(f"sanity_check, running_epochs changed to {running_epochs}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training device:", device)

    collator = DataCollatorWithPadding(tokenizer)

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator
    )

    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, collate_fn=collator
    )

    model = MultiHeadSigmoid(
        backbone_model_name,
        num_aspects=len(idx2topic),
        num_sentiments=len(idx2sentiment),
    ).to(device)

    print("Trainable layers:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    # alpha, counts = calculate_alpha(train_data, device=device)
    # print(f"Class counts: {counts}")
    # print(f"Focal loss alpha: {alpha}")
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(running_epochs):

        print(f"\nEpoch {epoch+1}")

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss = eval_epoch(model, val_loader, criterion, device)

        print("Train loss:", train_loss)
        print("Val loss:", val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "num_aspects": model.num_aspects,
                "num_sentiments": model.num_sentiments,
                "backbone_model_name": backbone_model_name,
            }
            torch.save(checkpoint, f"{best_model_name}.pth")
            print(f"Best {best_model_name} saved")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

    history = {"train_loss": train_losses, "val_loss": val_losses}
    history_file_name = "loss_history.json"
    with open(history_file_name, "w") as f:
        json.dump(history, f)

    if args.output_dir:
        upload_blob(f"{best_model_name}.pth", args.output_dir)
        upload_blob(history_file_name, args.output_dir)
        print(f"Training result SAVED {args.output_dir}!")
    else:
        print("Done training with NO SAVING")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output_dir", type=str, default=os.getenv("AIP_MODEL_DIR")
    )

    parser.add_argument(
        "--run_mode", type=str, default="train", help="train | sanity_check"
    )

    parser.add_argument("--epochs", type=int, default=EPOCHS)

    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)

    args = parser.parse_args()

    main(args)

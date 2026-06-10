import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from FCN_network import FullyConvNetwork
from facades_dataset import PairedImageDataset
from PIL import Image
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader


def tensor_to_image(tensor):
    """
    Convert a normalized tensor in [-1, 1] to an RGB uint8 image.
    """
    image = tensor.detach().cpu().clamp(-1, 1)
    image = (image + 1) / 2
    image = image.permute(1, 2, 0).numpy()
    return (image * 255).round().astype(np.uint8)


def save_images(inputs, targets, outputs, output_dir, split_name, epoch, num_images=5):
    """
    Save input/target/output comparison images for qualitative inspection.
    """
    folder = Path(output_dir) / split_name / f"epoch_{epoch + 1:03d}"
    folder.mkdir(parents=True, exist_ok=True)

    num_images = min(num_images, inputs.shape[0])
    for i in range(num_images):
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))
        Image.fromarray(comparison).save(folder / f"result_{i + 1}.png")


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, num_epochs, args):
    model.train()
    running_loss = 0.0

    for i, (image_input, image_target) in enumerate(dataloader):
        image_input = image_input.to(device)
        image_target = image_target.to(device)

        optimizer.zero_grad()
        outputs = model(image_input)

        if (epoch + 1) % args.sample_every == 0 and i == 0:
            save_images(image_input, image_target, outputs, args.results_dir, "train", epoch)

        loss = criterion(outputs, image_target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if (i + 1) % args.print_every == 0 or i == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Step [{i + 1}/{len(dataloader)}], "
                f"Loss: {loss.item():.4f}"
            )

    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device, epoch, args):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, (image_input, image_target) in enumerate(dataloader):
            image_input = image_input.to(device)
            image_target = image_target.to(device)
            outputs = model(image_input)

            loss = criterion(outputs, image_target)
            val_loss += loss.item()

            if (epoch + 1) % args.sample_every == 0 and i == 0:
                save_images(image_input, image_target, outputs, args.results_dir, "val", epoch)

    avg_val_loss = val_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{args.epochs}], Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss


def parse_args():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train the assignment Pix2Pix-style FCN.")
    parser.add_argument("--train-list", default=str(script_dir / "train_list.txt"))
    parser.add_argument("--val-list", default=str(script_dir / "val_list.txt"))
    parser.add_argument("--input-side", choices=["left", "right"], default="left")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--step-size", type=int, default=80)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--results-dir", default=str(script_dir / "results"))
    parser.add_argument("--checkpoint-dir", default=str(script_dir / "checkpoints"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = PairedImageDataset(
        list_file=args.train_list,
        image_size=args.image_size,
        input_side=args.input_side,
    )
    val_dataset = PairedImageDataset(
        list_file=args.val_list,
        image_size=args.image_size,
        input_side=args.input_side,
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = FullyConvNetwork().to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    results_dir = Path(args.results_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history_path = results_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_l1", "val_l1", "lr"])

        for epoch in range(args.epochs):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device, epoch, args.epochs, args
            )
            val_loss = validate(model, val_loader, criterion, device, epoch, args)
            current_lr = optimizer.param_groups[0]["lr"]
            writer.writerow([epoch + 1, f"{train_loss:.6f}", f"{val_loss:.6f}", current_lr])
            file.flush()

            scheduler.step()

            if (epoch + 1) % args.save_every == 0:
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_l1": val_loss,
                        "args": vars(args),
                    },
                    checkpoint_dir / f"pix2pix_fcn_epoch_{epoch + 1}.pth",
                )


if __name__ == "__main__":
    main()

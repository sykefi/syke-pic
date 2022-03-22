"""This module contains most of the logic for model training."""

import shutil
from pathlib import Path
from configparser import ConfigParser

import torch
from torch import optim, nn
from tqdm import tqdm
from sklearn.metrics import classification_report

from . import data, network
from .config import get_img_shape, get_transforms, get_network
from sykepic.analyze import plot


def main(args):

    config = ConfigParser()
    config.read(args.config)

    # [dataset]
    dataset = Path(config.get("dataset", "path"))
    split = tuple(float(i) for i in config.get("dataset", "split").split(","))
    min_N = config.get("dataset", "min_N")
    min_N = int(min_N) if min_N else None
    max_N = config.get("dataset", "max_N")
    max_N = int(max_N) if max_N else None
    exclude = [name.strip() for name in config.get("dataset", "exclude").split(",")]
    random_seed = config.getint("dataset", "random_seed")
    model_data = data.ModelData(dataset, split, min_N, max_N, exclude, random_seed)

    if args.save_images:
        extracted_images_root_dir = Path(args.save_images)
        (extracted_images_root_dir / "train").mkdir(exist_ok=True, parents=True)
        (extracted_images_root_dir / "test").mkdir(exist_ok=True)
        (extracted_images_root_dir / "val").mkdir(exist_ok=True)
        for img_path in model_data.train_x:
            shutil.copy(img_path, extracted_images_root_dir / "train" / img_path.name)
        for img_path in model_data.test_x:
            shutil.copy(img_path, extracted_images_root_dir / "test" / img_path.name)
        for img_path in model_data.val_x:
            shutil.copy(img_path, extracted_images_root_dir / "val" / img_path.name)

    # Create distribution plot and exit
    if args.dist:
        out_file = Path(args.dist)
        if not out_file.suffix:
            out_file = out_file.with_suffix(".png")
        plot.dataset_distribution(model_data, out_file)
        print(f"[INFO] Distribution plot saved to {out_file}")
        return

    if config.getboolean("dataset", "oversample"):
        decay = config.get("dataset", "oversample_decay")
        until = config.get("dataset", "oversample_until")
        if until:
            until = int(until)
        elif decay:
            decay = float(decay)
        model_data.oversample(until, decay)

    # [image]
    img_shape = get_img_shape(config)
    batch_size = config.getint("image", "batch_size")
    num_workers = config.getint("image", "num_workers")
    train_transform, eval_transform = get_transforms(config, img_shape)

    # Create training image collage and exit
    if args.collage:
        height, width, out_file = args.collage
        height = int(height)
        width = int(width)
        out_file = Path(out_file)
        batch_size = height * width
        model_data.set_data_loaders(
            batch_size,
            num_workers,
            train_transform,
            eval_transform,
            num_chans=img_shape[0],
        )
        if not out_file.suffix:
            out_file = out_file.with_suffix(".png")
        plot.view_batch(model_data.train_loader, height, width, out_file)
        print(f"[INFO] Image collage saved to {out_file}")
        return

    model_data.set_data_loaders(
        batch_size, num_workers, train_transform, eval_transform, num_chans=img_shape[0]
    )
    num_classes = len(model_data.le.classes_)

    # [model]
    model_id = config.get("model", "id")
    model_dir = Path(config.get("model", "path"))
    if model_id == "auto":
        model_id = data.auto_id(model_dir)
    model_dir = model_dir / f"version_{model_id}"
    model_dir.mkdir(parents=True, exist_ok=config.getboolean("model", "exist_ok"))
    # Save this model's training information
    model_data.save(model_dir)
    shutil.copy(args.config, model_dir / "config.ini")

    # [train]
    device = torch.device("cuda:0" if config.getboolean("train", "gpu") else "cpu")
    max_epochs = config.getint("train", "max_epochs")
    early_stop_patience = config.getint("train", "early_stop_patience")
    lr = config.getfloat("train", "learning_rate")
    optimizer = config.get("train", "optimizer")
    loss_fn = nn.CrossEntropyLoss()

    net = get_network(config, num_classes)
    network.freeze(net.base)
    initial_params = [p for p in net.parameters() if p.requires_grad]
    optimizer = getattr(optim, optimizer)(
        [
            {"params": initial_params, "lr": lr},
            {"params": [], "lr": 0.0},
            {"params": [], "lr": 0.0},
        ]
    )
    print("---- Network Head ----")
    print(net.head)

    # [lr_warmup]
    if config.getboolean("lr_warmup", "use"):
        f1 = config.getfloat("lr_warmup", "factor_1")
        f2 = config.getfloat("lr_warmup", "factor_2")
        s1 = config.getint("lr_warmup", "step_1")
        s2 = config.getint("lr_warmup", "step_2")
        s3 = config.getint("lr_warmup", "step_3")
        verbose = config.getboolean("lr_warmup", "verbose")
        lr_warmup = network.LRWarmup(net, optimizer, f1, f2, s1, s2, s3, verbose)
    else:
        lr_warmup = None

    # [lr_reduction]
    if config.getboolean("lr_reduction", "use"):
        factor = config.getfloat("lr_reduction", "factor")
        patience = config.getint("lr_reduction", "patience")
        verbose = config.getboolean("lr_reduction", "verbose")
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", factor, patience, verbose
        )
    else:
        lr_scheduler = None

    # Train and Evaluate
    best_state = train_net(
        net,
        model_data.train_loader,
        model_data.val_loader,
        optimizer,
        loss_fn,
        max_epochs,
        early_stop_patience,
        model_dir,
        device,
        lr_scheduler,
        lr_warmup,
    )
    net.load_state_dict(torch.load(best_state))
    test_report = test_net(net, model_data.test_loader, model_data.le.classes_, device)
    print(test_report)
    with open(model_dir / "test_report.txt", "w") as fh:
        fh.write(test_report)


def train_net(
    net,
    train_dataloader,
    val_dataloader,
    optimizer,
    loss_fn,
    max_epochs,
    early_stop_patience,
    model_dir,
    device,
    lr_scheduler=None,
    lr_warmup=None,
):

    net = net.to(device)
    max_val_acc = 0
    min_val_loss = 0
    no_improvement = 0
    train_accuracies = []
    train_losses = []
    val_accuracies = []
    val_losses = []
    best_state = model_dir / "best_state.pth"

    try:
        for epoch in range(1, max_epochs + 1):
            print(f"\n----- Epoch {epoch} -----")
            # Run callbacks
            if lr_warmup:
                lr_warmup(epoch)

            # Training phase
            net.train()
            train_acc = 0.0
            train_loss = 0.0
            num_samples = 0.0
            for batch in tqdm(train_dataloader):
                x, y = batch[0].to(device), batch[1].to(device)
                optimizer.zero_grad()
                out = net(x)
                loss = loss_fn(out, y)
                loss.backward()
                optimizer.step()
                preds = torch.argmax(out, dim=1)
                train_acc += torch.sum((preds == y), dtype=torch.float).item()
                # Loss is calculated per batch
                train_loss += loss.item() * len(y)
                num_samples += len(y)
            train_acc /= num_samples
            train_loss /= num_samples
            train_accuracies.append(train_acc)
            train_losses.append(train_loss)
            print(
                f"[STAT] Train Acc: {train_acc:.3f}, " f"Train Loss: {train_loss:.3f}"
            )

            # Validation Phase
            net.eval()
            val_acc = 0.0
            val_loss = 0.0
            num_samples = 0.0
            with torch.no_grad():
                for batch in val_dataloader:
                    x, y = batch[0].to(device), batch[1].to(device)
                    out = net(x)
                    loss = loss_fn(out, y)
                    preds = torch.argmax(out, dim=1)
                    val_acc += torch.sum((preds == y), dtype=torch.float).item()
                    val_loss += loss.item() * len(y)
                    num_samples += len(y)
            val_acc /= num_samples
            val_loss /= num_samples
            val_accuracies.append(val_acc)
            val_losses.append(val_loss)
            print(f"[STAT] Val Acc: {val_acc:.3f}, Val Loss: {val_loss:.3f}")

            # Checkpoint
            plot.plot_stats(
                train_accuracies,
                train_losses,
                val_accuracies,
                val_losses,
                outfile=model_dir / "train_stats.png",
                first_epoch=1,
                epoch_step=3,
            )
            if epoch >= 11:
                plot.plot_stats(
                    train_accuracies[10:],
                    train_losses[10:],
                    val_accuracies[10:],
                    val_losses[10:],
                    outfile=model_dir / "train_stats_zoomed.png",
                    first_epoch=11,
                    epoch_step=2,
                )
            if val_acc > max_val_acc:
                print("[INFO] Increased accuracy, saving model state")
                max_val_acc = val_acc
                torch.save(net.state_dict(), best_state)
            if val_loss < min_val_loss or epoch == 1:
                no_improvement = 0
                min_val_loss = val_loss
            else:
                no_improvement += 1
                print(f"[INFO] No reduction in loss for {no_improvement} epochs")
            if no_improvement >= early_stop_patience:
                print("[INFO] Stopping early")
                break
            if lr_scheduler:
                if not lr_warmup or epoch > lr_warmup.step_3:
                    lr_scheduler.step(val_loss)
    except KeyboardInterrupt:
        print("[INFO] Stopping early")

    except Exception as e:
        print(f"[ERROR] {e}")

    finally:
        return best_state


def test_net(net, dataloader, classes, device):
    # Testing Phase
    net = net.to(device)
    net.eval()
    test_acc = 0.0
    num_samples = 0.0
    print("\n----- Model Evaluation -----")
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch[0].to(device), batch[1].to(device)
            out = net(x)
            preds = torch.argmax(out, dim=1)
            test_acc += torch.sum((preds == y), dtype=torch.float).item()
            num_samples += len(y)
            true_labels.extend(y.tolist())
            predicted_labels.extend(preds.tolist())
    test_acc /= num_samples
    print(f"[STAT] Test Accuracy: {test_acc:.3f}\n")
    test_report = classification_report(
        true_labels, predicted_labels, target_names=classes
    )
    return test_report

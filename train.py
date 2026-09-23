import argparse
import torch
import time
import datetime
import yaml
import csv
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import matplotlib.pyplot as plt

from braindecode.models import EEGNetv4
from models import *
from util import (
    NakanishiHandler,
    BenchmarkHandler,
    UTECHandler,
    WearableHandlerDry,
    WearableHandlerWet,
    trainSubjectIndependent,
    seed_everything,
)
from util.engine import EarlyStopping, train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser(
        "SSVEPformer training and evaluation script", add_help=False
    )
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--epochs", default=100, type=int)

    # Model parameters
    parser.add_argument(
        "--model",
        default="ssvepformer",
        type=str,
        choices=[
            "ssvepformer",
            "f1ssvepformer_A",
            "f1ssvepformer_B",
            "f1ssvepformer_C",
            "f2ssvepformer_A",
            "f2ssvepformer_B",
            "f2ssvepformer_C",
            "EEGNet",
            "SSVEPNet",
            "Conformer",
            "Deformer",
        ],
        help="Name of model to train",
    )
    parser.add_argument(
        "--signal_size", default=1.0, type=float, help="signal sample size"
    )

    # Optimizer parameters
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
        help="weight decay (default: 1e-3)",
    )
    parser.add_argument(
        "--early-stopping",
        default=False,
        help="Actives early stopping in training",
        action="store_true",
    )
    parser.add_argument("--patience", default=20, type=int)

    parser.add_argument("--seed", default=1, type=int)

    # Dataset parameters
    parser.add_argument(
        "--data_path",
        default="datasets/2015_Nakanishi_SSVEP_database",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--params_path",
        default="datasets/Nakanishi.yaml",
        type=str,
        help="Parameter dataset path",
    )
    parser.add_argument(
        "--dataset",
        default="NAKANISHI",
        choices=["NAKANISHI", "BENCHMARK", "UTEC", "WEARABLE_DRY", "WEARABLE_WET"],
        type=str,
        help="Image Net dataset path",
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        help="path where to save, empty for no saving",
    )

    # Further config
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )

    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--lower_precision",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--persistent_workers",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--init_weights",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--pin-mem",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )

    parser.add_argument("--plot", action="store_true", default=False, help="")
    parser.add_argument("--save_model", action="store_true", default=False, help="")
    parser.set_defaults(pin_mem=True)
    return parser


def main(args):

    # Setup
    with open(args.params_path) as f:
        params = yaml.load(f, Loader=yaml.loader.SafeLoader)
    device = torch.device(args.device)
    seed_everything(args.seed)

    # Creata results file
    filename = os.path.join(
        args.output_dir, f"{args.model}_e{args.epochs}_{args.dataset}.csv"
    )
    if not (os.path.exists(filename)):
        with open(filename, "w") as file:
            writer = csv.writer(file)
            writer.writerow(["Time"] + [str(i + 1) for i in range(params["Subs"])])

    # Data handler choosing
    match args.dataset:
        case "NAKANISHI":
            datahandler = NakanishiHandler(params, args.signal_size, args.data_path)
        case "BENCHMARK":
            datahandler = BenchmarkHandler(params, args.signal_size, args.data_path)
        case "UTEC":
            datahandler = UTECHandler(params, args.signal_size, args.data_path)
        case "WEARABLE_WET":
            datahandler = WearableHandlerWet(params, args.signal_size, args.data_path)
        case "WEARABLE_DRY":
            datahandler = WearableHandlerDry(params, args.signal_size, args.data_path)
    train_x, train_y = datahandler.getAllSubjectsData()

    # Main training/validation and testing loop
    print(
        f"SSVEPformer: Training {args.model} net for {args.epochs} epochs/{args.batch_size} batch"
    )
    test_accuracy = []
    header = ["Time"] + list(range(1, params["Subs"] + 1))

    # Create a new row for this training run
    with open(filename, "a", newline="") as file:
        writer = csv.writer(file)

        # Add header if file doesn't exist or is empty
        if os.path.getsize(filename) == 0:
            writer.writerow(header)

        # Add an empty row for this run
        writer.writerow([args.signal_size])

    # Remember which row belongs to this run
    run_row = None

    # Find the last row (the one we just created)
    with open(filename, "r", newline="") as file:
        rows = list(csv.reader(file))
        run_row = len(rows) - 1

    for subject in range(params["Subs"]):
        data_loader_train, data_loader_val, data_loader_test = trainSubjectIndependent(
            train_x,
            train_y,
            subject + 1,
            params["Subs"],
            params["Trials"],
            params["Classes"],
            device,
            args,
            0.8,
            fs=params["Fs"],
        )
        # Build model and optimizer
        match args.model:
            case "ssvepformer":
                model = SSVEPformer(params, args.signal_size)
            case "f1ssvepformer_A":
                model = fuzzySSVEPformerA(params, args.signal_size)
            case "f1ssvepformer_B":
                model = fuzzySSVEPformerB(params, args.signal_size)
            case "f1ssvepformer_C":
                model = fuzzySSVEPformerC(params, args.signal_size)
            case "f2ssvepformer_A":
                model = fuzzyT2SSVEPformerA(params, args.signal_size)
            case "f2ssvepformer_B":
                model = fuzzyT2SSVEPformerB(params, args.signal_size)
            case "f2ssvepformer_C":
                model = fuzzyT2SSVEPformerC(params, args.signal_size)
            case "EEGNet":
                model = EEGNetv4(
                    params["Channels"],
                    params["Classes"],
                    int(args.signal_size * params["Fs"]) + 25,
                )
            case "SSVEPNet":
                model = ESNet(
                    params["Channels"],
                    round(args.signal_size * params["Fs"]) + 25,
                    params["Classes"],
                )
            case "Conformer":
                model = Conformer(
                    n_classes=params["Classes"],
                    n_chan=params["Channels"],
                    n_samples=round(args.signal_size * params["Fs"]),
                )
            case "Deformer":
                model = Deformer(
                    num_chan=params["Channels"],
                    num_time=round(args.signal_size * params["Fs"]) + 25,
                    temporal_kernel=11,
                    num_kernel=64,
                    num_classes=params["Classes"],
                    depth=4,
                    heads=16,
                    mlp_dim=16,
                    dim_head=16,
                    dropout=0.5,
                )

        if args.init_weights:
            model.apply(initialize_weights)
        model.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
        if args.early_stopping:
            early_stopping = EarlyStopping(patience=args.patience)

        # Loop
        start_time = time.time()
        train_losses = []
        valid_losses = []
        if args.compile:
            trained_model = torch.compile(model)
        else:
            trained_model = model
        print(f"Subject {subject + 1}")
        for epoch in range(args.epochs):
            train_loss = train_one_epoch(
                data_loader_train,
                trained_model,
                criterion,
                optimizer,
                device,
                args.lower_precision,
            )
            valid_loss, _ = evaluate(data_loader_val, trained_model, criterion, device)
            train_losses.append(train_loss / len(data_loader_train))
            valid_losses.append(valid_loss / len(data_loader_val))
            print(
                f"Epoch {epoch}/{args.epochs} / train loss:{train_losses[-1]:.4f} / val loss:{valid_losses[-1]:.4f}"
            )
            if args.early_stopping:
                early_stopping(valid_loss)
                if early_stopping.early_stop:
                    break

        if args.plot:
            plt.plot(train_losses)
            plt.plot(valid_losses)
            plt.savefig(
                f"plots/{args.model}_{args.dataset}_S{subject+1}_{args.signal_size}s.png"
            )
            plt.close()

        if args.save_model:
            # Check if folder exists, otherwise create it
            weights_path = "weights"
            if not os.path.exists(weights_path):
                os.makedirs(weights_path)

            # Use the actual training seed and keep the base name unchanged.
            checkpoint_stem = (
                f"{args.model}_{args.signal_size:.1f}s_{args.dataset}"
                f"_S{subject+1}_seed{args.seed}"
            )
            save_path = os.path.join(weights_path, f"{checkpoint_stem}.pth")

            # Preserve previous runs without accumulating suffixes.
            version = 2
            while os.path.exists(save_path):
                save_path = os.path.join(
                    weights_path, f"{checkpoint_stem}_run{version}.pth"
                )
                version += 1

            torch.save(model.state_dict(), save_path)

        # Reports
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))

        _, test_acc = evaluate(data_loader_test, trained_model, criterion, device)
        test_accuracy.append(round(test_acc, 4))
        print("Test accuracy {:.3f}".format(test_acc))
        print("")

        # Read existing CSV
        with open(filename, "r", newline="") as file:
            rows = list(csv.reader(file))

        # Update ONLY this run's row
        rows[run_row] = [args.signal_size] + test_accuracy

        # Rewrite CSV
        with open(filename, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "SSVEPformer training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)

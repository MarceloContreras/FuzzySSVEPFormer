import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml
import argparse
import numpy as np
import csv

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
from util.engine import get_latent_space, evaluate
from util.utils import find_checkpoint


def get_args_parser():
    parser = argparse.ArgumentParser("SSVEPformer evaluation script", add_help=False)

    # Model parameters
    parser.add_argument("--batch-size", default=128, type=int)
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
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )

    parser.add_argument("--save_latent", action="store_true", default=False, help="")

    parser.add_argument(
        "--persistent_workers",
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
    parser.set_defaults(pin_mem=True)
    return parser


def initialize_test_time_report(filename, signal_size, subjects, test_times, seeds):
    """Append one fixed-width row per test_time/repetition, preserving prior runs."""
    header = ["Time", "Test_time", "Repeat", "TimeSeed"] + [
        str(sub) for sub in subjects
    ]
    rows = []

    if os.path.exists(filename):
        with open(filename, "r", newline="") as file:
            rows = list(csv.reader(file))

    if rows and rows[0] != header:
        raise ValueError(
            f"Incompatible CSV header in {filename}. "
            "Move the old report or choose another output directory."
        )

    if any(len(row) != len(header) for row in rows):
        raise ValueError(
            f"Inconsistent row widths in {filename}; report left unchanged."
        )
    if not rows:
        rows = [header]

    run_rows = {}
    for seed_num, test_time_seed in enumerate(seeds):
        for test_time in test_times:
            run_rows[(test_time, seed_num)] = len(rows)
            rows.append(
                [signal_size, test_time, seed_num + 1, test_time_seed]
                + [""] * len(subjects)
            )

    subject_columns = {sub: 4 + i for i, sub in enumerate(subjects)}
    os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)

    with open(filename, "w", newline="") as file:
        csv.writer(file).writerows(rows)

    return rows, run_rows, subject_columns


def main(args):
    # Setup
    with open(args.params_path) as f:
        params = yaml.load(f, Loader=yaml.loader.SafeLoader)

    device = torch.device(args.device)

    # Variables to compare against
    WINDOWS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    SUBJECTS = list(range(1, params["Subs"] + 1))
    SEEDS = [1, 10, 30, 40, 50]

    # Build model and load weights
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
                round(args.signal_size * params["Fs"]),
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

    weights_path = "/home/marcelo/Documents/Tesis/run_pod_data/weights_nakanishi"
    criterion = torch.nn.CrossEntropyLoss()

    # Validate all checkpoint counts before creating report rows.
    checkpoints_by_subject = {
        sub: find_checkpoint(
            folder=weights_path,
            modelo=args.model,
            ventana=args.signal_size,
            dataset=args.dataset,
            sujeto=sub,
        )
        for sub in SUBJECTS
    }
    counts = {sub: len(paths) for sub, paths in checkpoints_by_subject.items()}
    if any(count != len(SEEDS) for count in counts.values()):
        raise ValueError(
            f"Expected {len(SEEDS)} checkpoints per subject; found {counts}. "
            "Repeat denotes sorted checkpoint position, not the training seed."
        )

    filename = os.path.join(
        args.output_dir, f"{args.model}_{args.dataset}_TimeVarying.csv"
    )
    rows, run_rows, subject_columns = initialize_test_time_report(
        filename, args.signal_size, SUBJECTS, WINDOWS, SEEDS
    )

    # Create folder to save embeddings
    if args.save_latent:
        if not os.path.exists("latent_space"):
            os.makedirs("latent_space")

    # * MAIN LOOP;
    for test_window_length in WINDOWS:
        for seed_num, seed in enumerate(SEEDS):
            seed_everything(seed)
            match args.dataset:
                case "NAKANISHI":
                    datahandler = NakanishiHandler(
                        params, test_window_length, args.data_path
                    )
                case "BENCHMARK":
                    datahandler = BenchmarkHandler(
                        params, test_window_length, args.data_path
                    )
                case "UTEC":
                    datahandler = UTECHandler(params, args.signal_size, args.data_path)
            train_x, train_y = datahandler.getAllSubjectsData()

            for sub in SUBJECTS:
                ckpt = checkpoints_by_subject[sub][seed_num]
                model.load_state_dict(torch.load(ckpt))
                model.to(device)

                _, _, data_loader_test = trainSubjectIndependent(
                    train_x,
                    train_y,
                    sub,
                    params["Subs"],
                    params["Trials"],
                    params["Classes"],
                    device,
                    args,
                    0.8,
                    fs=params["Fs"],
                )

                # * Accuracy
                _, test_acc = evaluate(data_loader_test, model, criterion, device)
                # Update the subject cell in this test_time/repetition row.
                row_index = run_rows[(test_window_length, seed_num)]
                rows[row_index][subject_columns[sub]] = round(test_acc, 4)
                with open(filename, "w", newline="") as file:
                    csv.writer(file).writerows(rows)

                # * Embeddings
                if args.save_latent:
                    X_latent, Y_latent_space = get_latent_space(
                        data_loader_test, model, device
                    )
                    np.save(
                        f"latent_space/X_{args.model}_{args.dataset}_S{sub+1}_t{test_window_length}_seed{seed}",
                        X_latent,
                    )
                    np.save(
                        f"latent_space/Y_{args.model}_{args.dataset}_S{sub+1}_t{test_window_length}_seed{seed}",
                        Y_latent_space,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "SSVEPformer training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)

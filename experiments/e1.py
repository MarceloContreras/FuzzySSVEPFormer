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
from util.itr import itr


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


def initialize_noise_report(filename, signal_size, subjects, sigmas, seeds):
    """Append one fixed-width row per noise/repetition, preserving prior runs."""
    header = ["Time", "Noise", "Repeat", "NoiseSeed"] + [str(sub) for sub in subjects]
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
    for seed_num, noise_seed in enumerate(seeds):
        for noise in sigmas:
            run_rows[(noise, seed_num)] = len(rows)
            rows.append(
                [signal_size, noise, seed_num + 1, noise_seed] + [""] * len(subjects)
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
    SIGMAS = [0, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
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

    match args.dataset:
        case "NAKANISHI":
            datahandler = NakanishiHandler(params, args.signal_size, args.data_path)
        case "BENCHMARK":
            datahandler = BenchmarkHandler(params, args.signal_size, args.data_path)
        case "UTEC":
            datahandler = UTECHandler(params, args.signal_size, args.data_path)

    train_x, train_y = datahandler.getAllSubjectsData()
    weights_path = "/home/marcelo/Documents/Tesis/results/checkpoints/utec"
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

    filename = os.path.join(args.output_dir, f"{args.model}_{args.dataset}_Noise.csv")
    rows, run_rows, subject_columns = initialize_noise_report(
        filename, args.signal_size, SUBJECTS, SIGMAS, SEEDS
    )

    itr_filename = os.path.join(
        args.output_dir, f"{args.model}_{args.dataset}_Noise_ITR.csv"
    )
    itr_rows, itr_run_rows, itr_subject_columns = initialize_noise_report(
        itr_filename, args.signal_size, SUBJECTS, SIGMAS, SEEDS
    )

    # Create folder to save embeddings
    if args.save_latent:
        if not os.path.exists("latent_space"):
            os.makedirs("latent_space")

    # * MAIN LOOP
    for sub in SUBJECTS:
        checkpoints = checkpoints_by_subject[sub]

        for seed_num, ckpt in enumerate(checkpoints):
            model.load_state_dict(torch.load(ckpt))
            model.to(device)

            for noise in SIGMAS:
                # * Pollute data with noise
                rng = np.random.default_rng(seed=SEEDS[seed_num])
                train_x_noisy = train_x + rng.normal(0, noise, train_x.shape)
                _, _, data_loader_test = trainSubjectIndependent(
                    train_x_noisy,
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
                # Update the subject cell in this noise/repetition row.
                row_index = run_rows[(noise, seed_num)]
                rows[row_index][subject_columns[sub]] = round(test_acc, 4)
                with open(filename, "w", newline="") as file:
                    csv.writer(file).writerows(rows)

                # ITR in bits/min; evaluate returns accuracy as a percentage.
                test_itr = itr(test_acc / 100.0, params["Classes"], args.signal_size)
                itr_row_index = itr_run_rows[(noise, seed_num)]
                itr_rows[itr_row_index][itr_subject_columns[sub]] = round(test_itr, 4)
                with open(itr_filename, "w", newline="") as file:
                    csv.writer(file).writerows(itr_rows)

                # * Embeddings
                if args.save_latent:
                    X_latent, Y_latent_space = get_latent_space(
                        data_loader_test, model, device
                    )
                    np.save(
                        f"/home/marcelo/Documents/Tesis/results/embeddings/e1/{args.dataset}/X_{args.model}_S{sub}_n{noise}_seed{seed_num}",
                        X_latent,
                    )
                    np.save(
                        f"/home/marcelo/Documents/Tesis/results/embeddings/e1/{args.dataset}/Y_{args.model}_S{sub}_n{noise}_seed{seed_num}",
                        Y_latent_space,
                    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "SSVEPformer training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)

import os
import numpy as np
import scipy.io as sio
import argparse
import csv

from sklearn.metrics import confusion_matrix
from scripts import ssvep_utils as su
from meegkit.trca import TRCA


def get_args_parser():
    parser = argparse.ArgumentParser(
        "CCNN training and evaluation script", add_help=False
    )
    parser.add_argument(
        "--signal_size", default=1.0, type=float, help="signal sample size"
    )
    # Dataset parameters
    parser.add_argument(
        "--dataset_path",
        default="datasets/Noisy-UTEC",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        help="path where to save, empty for no saving",
    )
    parser.set_defaults(pin_mem=True)
    return parser


all_segment_data = dict()
all_acc = list()
n_classes = 4
sample_rate = 250
flicker_freq = np.array([7, 8, 9, 10])


def get_subject_indepedent(
    target_subject: int,
    window_len: float = 1.0,
    shift_len: float = 1.0,
    data_path: str = None,
):

    dataset = sio.loadmat(f"{data_path}/S{target_subject+1}.mat")
    eeg = np.array(dataset["eeg"], dtype="float32")
    eeg = np.swapaxes(eeg, 1, 2)
    eeg = np.swapaxes(eeg, 2, 3)

    num_classes = eeg.shape[0]
    n_ch = eeg.shape[1]
    total_trial_len = eeg.shape[2]
    num_trials = eeg.shape[3]

    filtered_data = su.get_filtered_eeg(
        eeg, 4, 30, 4, sample_rate, onset=250, delay=0.135, signal_end=3.5
    )
    segmented_data = su.get_segmented_epochs(
        filtered_data, window_len, shift_len, sample_rate
    )
    segmented_data = np.expand_dims(segmented_data[:, :, :, 0, :], axis=3)
    segmented_data = np.swapaxes(segmented_data, 1, 3)

    # Finally reshaping the data into dim [classes*trials*segments X channels X features X 1]
    train_data = segmented_data.reshape(
        segmented_data.shape[0] * segmented_data.shape[1] * segmented_data.shape[2],
        segmented_data.shape[3],
        segmented_data.shape[4],
    )
    dummy_train = np.random.normal(
        0,
        1,
        (
            segmented_data.shape[0] * segmented_data.shape[1] * segmented_data.shape[2],
            segmented_data.shape[3],
            20,
        ),
    )
    train_data = np.concatenate((train_data, dummy_train), axis=-1)

    # Deberian ser [0*trials*sessions,1*trials*sessions]
    labels = np.repeat(
        np.arange(num_classes), segmented_data.shape[1] * segmented_data.shape[2]
    )

    # Defining data
    test_X = train_data.T
    test_Y = labels

    count = 0
    for subject in np.arange(0, 10):
        if subject == target_subject:
            continue
        dataset = sio.loadmat(f"{data_path}/S{subject+1}.mat")
        eeg = np.array(dataset["eeg"], dtype="float32")
        eeg = np.swapaxes(eeg, 1, 2)
        eeg = np.swapaxes(eeg, 2, 3)

        num_classes = eeg.shape[0]
        n_ch = eeg.shape[1]
        total_trial_len = eeg.shape[2]
        num_trials = eeg.shape[3]

        filtered_data = su.get_filtered_eeg(
            eeg, 4, 30, 4, sample_rate, onset=250, delay=0.135, signal_end=3.5
        )
        segmented_data = su.get_segmented_epochs(
            filtered_data, window_len, shift_len, sample_rate
        )
        segmented_data = np.expand_dims(segmented_data[:, :, :, 0, :], axis=3)
        segmented_data = np.swapaxes(segmented_data, 1, 3)

        # Finally reshaping the data into dim [classes*trials*segments X channels X features X 1]
        train_data = segmented_data.reshape(
            segmented_data.shape[0] * segmented_data.shape[1] * segmented_data.shape[2],
            segmented_data.shape[3],
            segmented_data.shape[4],
        )
        dummy_train = np.random.normal(
            0,
            1,
            (
                segmented_data.shape[0]
                * segmented_data.shape[1]
                * segmented_data.shape[2],
                segmented_data.shape[3],
                20,
            ),
        )
        train_data = np.concatenate((train_data, dummy_train), axis=-1)

        # Deberian ser [0*trials*sessions,1*trials*sessions]
        labels = np.repeat(
            np.arange(num_classes), segmented_data.shape[1] * segmented_data.shape[2]
        )

        if count == 0:
            train_X = train_data.T
            train_Y = labels
        else:
            # Merging arrays
            train_X = np.concatenate((train_X, train_data.T), axis=-1)
            train_Y = np.concatenate((train_Y, labels), axis=-1)

        count = count + 1

    return train_X, train_Y, test_X, test_Y


def main(args):

    test_accuracy = []
    num_subjects = 10
    # Creata results file
    filename = os.path.join(args.output_dir, f"TRCA_UTEC.csv")
    if not (os.path.exists(filename)):
        with open(filename, "w") as file:
            writer = csv.writer(file)
            writer.writerow(["Time"] + [str(i + 1) for i in range(num_subjects)])

    for i in range(10):
        is_ensemble = True
        sfreq = 250
        filterbank = [
            [(6, 90), (4, 100)],  # passband, stopband freqs [(Wp), (Ws)]
            [(14, 90), (10, 100)],
            [(22, 90), (16, 100)],
            [(30, 90), (24, 100)],
            [(38, 90), (32, 100)],
            [(46, 90), (40, 100)],
            [(54, 90), (48, 100)],
        ]

        trca = TRCA(sfreq, filterbank, is_ensemble)

        train_X, train_Y, test_X, test_Y = get_subject_indepedent(
            i, args.signal_size, args.signal_size, args.dataset_path
        )

        # Train
        trca.fit(train_X, train_Y)

        # Test
        estimated = trca.predict(test_X)

        # Evaluation of the performance for this fold (accuracy and ITR)
        is_correct = estimated == test_Y
        accs = np.mean(is_correct) * 100
        test_accuracy.append(accs)

    # Writing results
    with open(filename, "a") as file:
        writer = csv.writer(file)
        writer.writerow([args.signal_size] + test_accuracy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "SSVEPformer training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)

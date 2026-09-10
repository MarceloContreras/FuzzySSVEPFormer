import os
import csv
import numpy as np
import scipy.io as sio
import argparse

from keras.utils import to_categorical
from keras import optimizers
from keras.losses import categorical_crossentropy
from scripts import ssvep_utils as su


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
        default="/home/marcelo/Documents/Tesis/FuzzySSVEPFormer/datasets/Noisy-UTEC",
        type=str,
        help="dataset path",
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        help="path where to save, empty for no saving",
    )
    parser.add_argument("--save_model", action="store_true", default=False, help="")
    parser.add_argument(
        "--model_output",
        default="weights",
        help="path where to save, empty for no saving",
    )
    parser.set_defaults(pin_mem=True)
    return parser


CNN_PARAMS = {
    "batch_size": 64,
    "epochs": 50,
    "droprate": 0.25,
    "learning_rate": 0.001,
    "lr_decay": 0.0,
    "l2_lambda": 0.0001,
    "momentum": 0.9,
    "kernel_f": 10,
    "n_ch": 3,
    "num_classes": 4,
}

FFT_PARAMS = {
    "resolution": 0.5,
    "start_frequency": 3.0,
    "end_frequency": 35.0,
    "sampling_rate": 250,
}


def main(args):

    num_subjects = 10
    test_accuracy = []

    # Creata results file
    filename = os.path.join(args.output_dir, f"CNN_e{CNN_PARAMS['epochs']}_UTEC.csv")
    if not (os.path.exists(filename)):
        with open(filename, "w") as file:
            writer = csv.writer(file)
            writer.writerow(["Time"] + [str(i + 1) for i in range(num_subjects)])

    window_length = args.signal_size

    for subject in range(0, num_subjects):
        all_subjects_train_X = []
        all_subjects_train_Y = []

        for non_target_sub in range(0, num_subjects):
            if subject == non_target_sub:
                continue

            dataset = sio.loadmat(f"{args.dataset_path}/S{non_target_sub+1}.mat")
            eeg = np.array(dataset["eeg"], dtype="float32")
            eeg = np.swapaxes(eeg, 1, 2)
            eeg = np.swapaxes(eeg, 2, 3)

            CNN_PARAMS["num_classes"] = eeg.shape[0]
            CNN_PARAMS["n_ch"] = eeg.shape[1]
            sample_rate = 256

            filtered_data = su.get_filtered_eeg(
                eeg, 4, 30, 4, sample_rate, onset=250, delay=0.135, signal_end=3.5
            )
            eeg = []

            window_len = window_length
            shift_len = window_length

            segmented_data = su.get_segmented_epochs(
                filtered_data, window_len, shift_len, sample_rate
            )
            segmented_data = np.expand_dims(segmented_data[:, :, :, 0, :], axis=3)
            filtered_data = []

            features_data = su.complex_spectrum_features(segmented_data, FFT_PARAMS)
            segmented_data = []

            # Combining the features into a matrix of dim [features X channels X classes X trials*segments]
            features_data = np.reshape(
                features_data,
                (
                    features_data.shape[0],
                    features_data.shape[1],
                    features_data.shape[2],
                    features_data.shape[3] * features_data.shape[4],
                ),
            )

            train_data = features_data[:, :, 0, :].T

            # Reshaping the data into dim [classes*trials*segments X channels X features]
            for target in range(1, features_data.shape[2]):
                train_data = np.vstack(
                    [train_data, np.squeeze(features_data[:, :, target, :]).T]
                )

            # Finally reshaping the data into dim [classes*trials*segments X channels X features X 1]
            train_data = np.reshape(
                train_data,
                (train_data.shape[0], train_data.shape[1], train_data.shape[2], 1),
            )

            total_epochs_per_class = features_data.shape[3]
            features_data = []

            class_labels = np.arange(CNN_PARAMS["num_classes"])
            labels = (np.tile(class_labels, (total_epochs_per_class, 1)).T).ravel()
            labels = to_categorical(labels)

            # Merging subject train part
            all_subjects_train_X.append(train_data)
            all_subjects_train_Y.append(labels)

        all_subjects_train_X = np.array(all_subjects_train_X)
        new_shape = (
            all_subjects_train_X.shape[0] * all_subjects_train_X.shape[1],
            all_subjects_train_X.shape[2],
            all_subjects_train_X.shape[3],
            1,
        )
        all_subjects_train_X = all_subjects_train_X.reshape(new_shape)

        all_subjects_train_Y = np.array(all_subjects_train_Y)
        new_shape = (
            all_subjects_train_Y.shape[0] * all_subjects_train_Y.shape[1],
            all_subjects_train_Y.shape[2],
        )
        all_subjects_train_Y = np.reshape(all_subjects_train_Y, new_shape)

        ## Test part
        dataset = sio.loadmat(f"{args.dataset_path}/S{subject+1}.mat")
        eeg = np.array(dataset["eeg"], dtype="float32")
        eeg = np.swapaxes(eeg, 1, 2)
        eeg = np.swapaxes(eeg, 2, 3)

        CNN_PARAMS["num_classes"] = eeg.shape[0]
        CNN_PARAMS["n_ch"] = eeg.shape[1]
        sample_rate = 256

        filtered_data = su.get_filtered_eeg(
            eeg, 4, 30, 4, sample_rate, onset=250, delay=0.135, signal_end=3.5
        )
        eeg = []

        window_len = window_length
        shift_len = window_length

        segmented_data = su.get_segmented_epochs(
            filtered_data, window_len, shift_len, sample_rate
        )
        segmented_data = np.expand_dims(segmented_data[:, :, :, 0, :], axis=3)
        filtered_data = []

        features_data = su.complex_spectrum_features(segmented_data, FFT_PARAMS)
        segmented_data = []

        # Combining the features into a matrix of dim [features X channels X classes X trials*segments]
        features_data = np.reshape(
            features_data,
            (
                features_data.shape[0],
                features_data.shape[1],
                features_data.shape[2],
                features_data.shape[3] * features_data.shape[4],
            ),
        )

        test_data = features_data[:, :, 0, :].T

        # Reshaping the data into dim [classes*trials*segments X channels X features]
        for target in range(1, features_data.shape[2]):
            test_data = np.vstack(
                [test_data, np.squeeze(features_data[:, :, target, :]).T]
            )

        # Finally reshaping the data into dim [classes*trials*segments X channels X features X 1]
        all_subjects_test_X = np.reshape(
            test_data, (test_data.shape[0], test_data.shape[1], test_data.shape[2])
        )

        total_epochs_per_class = features_data.shape[3]
        features_data = []

        class_labels = np.arange(CNN_PARAMS["num_classes"])
        labels = (np.tile(class_labels, (total_epochs_per_class, 1)).T).ravel()
        all_subjects_test_Y = to_categorical(labels)

        x_tr, x_ts = all_subjects_train_X, all_subjects_test_X
        y_tr, y_ts = all_subjects_train_Y, all_subjects_test_Y
        input_shape = np.array([x_tr.shape[1], x_tr.shape[2], x_tr.shape[3]])

        # Actual training and evaluation steps
        model = su.CNN_model(input_shape, CNN_PARAMS)
        sgd = optimizers.SGD(
            learning_rate=CNN_PARAMS["learning_rate"],
            decay=CNN_PARAMS["lr_decay"],
            momentum=CNN_PARAMS["momentum"],
            nesterov=False,
        )
        model.compile(
            loss=categorical_crossentropy, optimizer=sgd, metrics=["accuracy"]
        )
        _ = model.fit(
            x_tr,
            y_tr,
            batch_size=CNN_PARAMS["batch_size"],
            epochs=CNN_PARAMS["epochs"],
            verbose=0,
        )
        score = model.evaluate(x_ts, y_ts, verbose=0)
        test_accuracy.append(round(score[1] * 100, 4))

        # Save model
        if args.save_model:
            # Check if folder exists, otherwise create it
            weights_path = "weights"
            os.makedirs(weights_path, exist_ok=True)

            # Base model name
            base_name = f"CNN_{args.signal_size:.1f}s_" f"UTEC_S{subject+1}"

            # First version
            save_path = os.path.join(weights_path, f"{base_name}.weights.h5")

            # Find next available version
            version = 2
            while os.path.exists(save_path):
                save_path = os.path.join(
                    weights_path, f"{base_name}_v{version}.weights.h5"
                )
                version += 1

            # Save weights
            model.save_weights(save_path)

            print(f"Model weights saved to: {save_path}")

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

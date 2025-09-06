import scipy.io as sio
import os
import math
import numpy as np
from scipy.signal import butter, iirnotch, filtfilt


class DataHandler(object):
    def __init__(self, params, time_window, path):
        self.fs = params["Fs"]
        self.path = path
        self.trials = params["Trials"]
        self.classes = params["Classes"]
        self.num_chn = params["Channels"]
        self.filt_axis = params["Filt.axis"]
        self.filt_order = params["Filt.order"]
        self.lowfreq_cut = params["Filt.low_cut"]
        self.trigger_est = params["Trigger"]
        self.num_samples = round(time_window * self.fs)
        self.time_window = time_window
        self.highfreq_cut = params["Filt.high_cut"]
        self.num_subjects = params["Subs"]

    def getSubjectData(self, num_sub):
        try:
            sub_path = os.path.join(self.path, "s" + str(num_sub) + ".mat")
            loaded_mat = sio.loadmat(sub_path)
        except:
            sub_path = os.path.join(self.path, "S" + str(num_sub) + ".mat")
            loaded_mat = sio.loadmat(sub_path)

        try:
            data = loaded_mat["eeg"]
        except:
            data = loaded_mat["data"]
        return data

    def filterSignal(self, data):
        nyq = 0.5 * self.fs
        low = self.lowfreq_cut / nyq
        high = self.highfreq_cut / nyq
        b, a = butter(self.filt_order, [low, high], btype="band")
        y = filtfilt(b, a, data, axis=self.filt_axis)
        return y

    def sliceTemporal(self, data):
        """
        Slices the signal to temporal section of [0.135, d + 0.135]
        where d is the window size (in seconds). The estimulus
        triggering was determined in 0.135
        """
        return data[:, :, : round(self.time_window * self.fs), :]

    def reorderEEGmatrix(self, data):
        """
        Modifies the EEG mat to fit for network input.
        Assumes a initial shape of (subject,class,chn,samples,trial)
        and gives an array-alike (subject,class,trial,chn,samples)
        """
        x = np.swapaxes(data, 2, 4)  # Put the trials'axis before the EEG matrix
        x = np.swapaxes(x, 3, 4)  # Put the EEG in the shape (chn,samples)
        return x

    def getAllSubjectsData(self):

        base_y = np.mgrid[0 : self.classes, 0 : self.trials][0].flatten()

        # Get train data
        train_x = np.zeros(
            (
                self.num_subjects,
                self.classes,
                self.num_chn,
                self.num_samples,
                self.trials,
            )
        )
        for sub in range(self.num_subjects):
            sub_signal = self.getSubjectData(sub + 1)
            sub_signal = sub_signal[
                :,
                :,
                int(39 + self.trigger_est * self.fs) : int(
                    38 + self.trigger_est * self.fs + 3 * self.fs - 1
                ),
                :,
            ]  # Onset + visual delay
            preproc_signal = self.filterSignal(sub_signal)
            preproc_signal = self.sliceTemporal(preproc_signal)
            preproc_signal = np.expand_dims(preproc_signal, axis=0)
            train_x[sub, ...] = preproc_signal

        train_x = self.reorderEEGmatrix(train_x)
        dummy_train = np.random.normal(0, 1, train_x.shape)
        train_x = np.concatenate((train_x, dummy_train[..., :25]), axis=-1)

        train_y = np.tile(base_y, self.num_subjects)

        return train_x, train_y


class NakanishiHandler(DataHandler):
    def __init__(self, params, time_window, path):
        super().__init__(params, time_window, path)


class UTECHandler(DataHandler):
    def __init__(self, params, time_window, path):
        super().__init__(params, time_window, path)

    def sliceTemporal(self, data):
        """
        Slices the signal to temporal section of [0.135, d + 0.135]
        where d is the window size (in seconds). The estimulus
        triggering was determined in 0.135
        """
        return data[:, :, :, : round(self.time_window * self.fs)]

    def getAllSubjectsData(self):

        base_y = np.mgrid[0 : self.classes, 0 : self.trials][0].flatten()

        # Get train data
        train_x = np.zeros(
            (
                self.num_subjects,
                self.classes,
                self.trials,
                self.num_chn,
                self.num_samples,
            )
        )
        for sub in range(self.num_subjects):
            sub_signal = self.getSubjectData(sub + 1)
            sub_signal = sub_signal[
                :,
                :,
                :,
                int(250 + self.trigger_est * self.fs) : int(
                    250 + self.trigger_est * self.fs + 4 * self.fs - 1
                ),
            ]  # Onset + visual delay
            preproc_signal = self.filterSignal(sub_signal)
            preproc_signal = self.sliceTemporal(preproc_signal)
            preproc_signal = np.expand_dims(preproc_signal, axis=0)
            train_x[sub, ...] = preproc_signal

        dummy_train = np.random.normal(0, 1, train_x.shape)
        train_x = np.concatenate((train_x, dummy_train[..., :25]), axis=-1)

        train_y = np.tile(base_y, self.num_subjects)
        return train_x, train_y


class BenchmarkHandler(DataHandler):
    def __init__(self, params, time_window, path):
        super().__init__(params, time_window, path)

    def reorderEEGmatrix(self, data):
        """
        Modifies the EEG mat to fit for network input.
        Assumes a initial shape of (subject,chn,samples,classes,trial)
        """
        x = np.swapaxes(data, 1, 3)
        x = np.swapaxes(x, 2, 4)
        chn_list = [47, 53, 54, 55, 56, 57, 60, 61, 62]
        return x[:, :, :, chn_list, :]

    def sliceTemporal(self, data):
        """
        Slices the signal to temporal section of [0.64, d + 0.64]
        where d is the window size (in seconds). The estimulus
        triggering was determined in 0.64
        """
        return data[:, : round(self.time_window * self.fs), :, :]

    def getAllSubjectsData(self):
        base_y = np.mgrid[0 : self.classes, 0 : self.trials][0].flatten()
        # Get train data
        train_x = np.zeros(
            (self.num_subjects, 64, self.num_samples, self.classes, self.trials)
        )
        for sub in range(self.num_subjects):
            sub_signal = self.getSubjectData(sub + 1)
            sub_signal = sub_signal[
                :,
                int(125 + self.trigger_est * self.fs) : int(
                    125 + self.trigger_est * self.fs + 4.5 * self.fs - 1
                ),
                :,
                :,
            ]  # Onset + visual delay
            preproc_signal = self.filterSignal(sub_signal)
            preproc_signal = self.sliceTemporal(preproc_signal)
            preproc_signal = np.expand_dims(preproc_signal, axis=0)
            train_x[sub, ...] = preproc_signal

        train_x = self.reorderEEGmatrix(train_x)
        train_y = np.tile(base_y, self.num_subjects)

        return train_x, train_y


class WearableHandlerWet(DataHandler):
    def __init__(self, params, time_window, path):
        super().__init__(params, time_window, path)

    def getSubjectData(self, num_sub):

        if num_sub < 10:
            file_name = "S00{:n}.mat".format(num_sub)
        elif num_sub < 100:
            file_name = "S0{:n}.mat".format(num_sub)
        else:
            file_name = "S{:n}.mat".format(num_sub)

        sub_path = os.path.join(self.path, file_name)
        loaded_mat = sio.loadmat(sub_path)

        return loaded_mat["data"][:, :, 1, :, :]  # Taking Wet electrodes only

    def reorderEEGmatrix(self, data):
        """
        Modifies the EEG mat to fit for network input.
        Assumes a initial shape of (subject,chn,samples,trial,classes)
        and gives an array-alike (subject,class,trial,chn,samples)
        """
        x = np.swapaxes(data, 1, 4)
        x = np.swapaxes(x, 2, 3)
        x = np.swapaxes(x, 3, 4)
        return x

    def sliceTemporal(self, data):
        """
        Slices the signal to temporal section of [0.64, d + 0.64]
        where d is the window size (in seconds). The estimulus
        triggering was determined in 0.64
        """
        return data[:, : round(self.time_window * self.fs), :, :]

    def getAllSubjectsData(self):
        base_y = np.mgrid[0 : self.classes, 0 : self.trials][0].flatten()
        # Get train data
        train_x = np.zeros(
            (
                self.num_subjects,
                self.num_chn,
                self.num_samples,
                self.trials,
                self.classes,
            )
        )
        for sub in range(self.num_subjects):
            sub_signal = self.getSubjectData(sub + 1)
            sub_signal = sub_signal[
                :,
                int(125 + self.trigger_est * self.fs) : int(
                    125 + self.trigger_est * self.fs + 2.0 * self.fs
                ),
                :,
                :,
            ]  # Onset + visual delay
            preproc_signal = self.filterSignal(sub_signal)
            preproc_signal = self.sliceTemporal(preproc_signal)
            preproc_signal = np.expand_dims(preproc_signal, axis=0)
            train_x[sub, ...] = preproc_signal

        train_x = self.reorderEEGmatrix(train_x)
        train_y = np.tile(base_y, self.num_subjects)

        return train_x, train_y


class WearableHandlerDry(WearableHandlerWet):
    def __init__(self, params, time_window, path):
        super().__init__(params, time_window, path)

    def getSubjectData(self, num_sub):

        if num_sub < 10:
            file_name = "S00{:n}.mat".format(num_sub)
        elif num_sub < 100:
            file_name = "S0{:n}.mat".format(num_sub)
        else:
            file_name = "S{:n}.mat".format(num_sub)

        sub_path = os.path.join(self.path, file_name)
        loaded_mat = sio.loadmat(sub_path)

        return loaded_mat["data"][:, :, 0, :, :]  # Taking dry electrodes only

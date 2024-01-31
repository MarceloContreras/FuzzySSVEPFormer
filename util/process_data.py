import scipy.io as sio
import os
import math
import numpy as np
from scipy.signal import butter, lfilter, iirnotch, filtfilt

class DataHandler(object):
    def __init__(self,params,time_window,path):
        self.fs = params['Fs']
        self.path = path
        self.trials = params['Trials']
        self.classes = params['Classes']
        self.num_chn = params['Channels']
        self.filt_axis = params['Filt.axis']
        self.filt_order = params['Filt.order']
        self.lowfreq_cut = params['Filt.low_cut']
        self.trigger_est = params['Trigger']
        self.num_samples = int(time_window*self.fs)
        self.time_window = time_window
        self.highfreq_cut = params['Filt.high_cut']
        self.num_subjects = params['Subs']
        self.check_length = params['Check_length']

    def getSubjectData(self,num_sub):
        try:
            sub_path = os.path.join(self.path,'s'+ str(num_sub)+'.mat')
            loaded_mat = sio.loadmat(sub_path)
        except:
            sub_path = os.path.join(self.path,'S'+ str(num_sub)+'.mat')
            loaded_mat = sio.loadmat(sub_path)
        
        try:
            data = loaded_mat['eeg']
        except: 
            data = loaded_mat['data']
        return data
    
    def filterSignal(self,data):
        nyq = 0.5 * self.fs
        low = self.lowfreq_cut / nyq
        high = self.highfreq_cut / nyq
        b, a = butter(self.filt_order, [low, high], btype='band')
        y = lfilter(b, a, data, axis = self.filt_axis)
        return y
    
    def sliceTemporal(self,data):
        """
        Slices the signal to temporal section of [0.135, d + 0.135]
        where d is the window size (in seconds). The estimulus
        triggering was determined in 0.135
        """
        low_bound   = int(self.trigger_est*self.fs)
        high_bound  = int((self.trigger_est+self.time_window)*self.fs)
        if self.time_window in self.check_length:
            low_bound   = math.ceil(self.trigger_est*self.fs)
        else:
            low_bound   = int(self.trigger_est*self.fs)
        return data[:,:,low_bound:high_bound,:]
    
    def reorderEEGmatrix(self,data):
        """
        Modifies the EEG mat to fit for network input.
        Assumes a initial shape of (subject,class,chn,samples,trial)
        and gives an array-alike (subject,class,trial,chn,samples)
        """
        x = np.swapaxes(data,2,4) # Put the trials'axis before the EEG matrix
        x = np.swapaxes(x,3,4) # Put the EEG in the shape (chn,samples)
        #x = x.reshape((-1,self.num_chn,self.num_samples)) # Reshapes it to merge subject,trials & estimulus
        return x
    
    def getAllSubjectsData(self):

        base_y  = np.mgrid[0:self.classes,0:self.trials][0].flatten()

        # Get train data
        train_x = np.zeros((self.num_subjects,self.classes,self.num_chn,self.num_samples,self.trials))
        for sub in range(self.num_subjects):
            sub_signal = self.getSubjectData(sub + 1)
            preproc_signal = self.filterSignal(sub_signal)
            preproc_signal = self.sliceTemporal(preproc_signal)
            preproc_signal = np.expand_dims(preproc_signal,axis = 0)
            train_x[sub,...] = preproc_signal

        train_x = self.reorderEEGmatrix(train_x)
        train_y = np.tile(base_y,self.num_subjects)

        return train_x,train_y


class NakanishiHandler(DataHandler):
    def __init__(self, params, time_window, path):
        super().__init__(params, time_window, path)    
    
class UTECHandler(DataHandler):
    def __init__(self, params, time_window, path):
        super().__init__(params, time_window, path)

    def filterSignal(self,data):
        # Notch filterind to remove signal noise 60 Hz
        b_notch, a_notch = iirnotch(60.0, 20.0, self.fs)
        y = filtfilt(b_notch, a_notch, data, axis = self.filt_axis)
        # Butter bandpass filter
        nyq = 0.5 * self.fs
        low = self.lowfreq_cut / nyq
        high = self.highfreq_cut / nyq
        b, a = butter(self.filt_orderorder, [low, high], btype='band')
        y = lfilter(b, a, y, axis = self.filt_axis)
        return y

class BenchmarkHandler(DataHandler):
    def __init__(self, params, time_window, path):
        super().__init__(params, time_window, path)

    def reorderEEGmatrix(self,data):
        """
        Modifies the EEG mat to fit for network input.
        Assumes a initial shape of (subject,chn,samples,classes,trial)
        """
        x = np.swapaxes(data,1,3)
        x = np.swapaxes(x,2,4)
        #x = x.reshape((-1,64,self.num_samples))
        chn_list = [47,53,54,55,56,57,60,61,62]
        return x[:,:,:,chn_list,:]

    def sliceTemporal(self,data):
        """
        Slices the signal to temporal section of [0.64, d + 0.64]
        where d is the window size (in seconds). The estimulus
        triggering was determined in 0.64
        """
        low_bound   = int(self.trigger_est*self.fs)
        high_bound  = int((self.trigger_est+self.time_window)*self.fs)
        if self.time_window in self.check_length:
            low_bound   = math.ceil(self.trigger_est*self.fs)
        else:
            low_bound   = int(self.trigger_est*self.fs)
        return data[:,low_bound:high_bound,:,:]

    def getAllSubjectsData(self):

        base_y  = np.mgrid[0:self.classes,0:self.trials][0].flatten()

        # Get train data
        train_x = np.zeros((self.num_subjects,64,self.num_samples,self.classes,self.trials))
        for sub in range(self.num_subjects):
            sub_signal = self.getSubjectData(sub + 1)
            preproc_signal = self.filterSignal(sub_signal)
            preproc_signal = self.sliceTemporal(preproc_signal)
            preproc_signal = np.expand_dims(preproc_signal,axis = 0)
            train_x[sub,...] = preproc_signal

        train_x = self.reorderEEGmatrix(train_x)
        train_y = np.tile(base_y,self.num_subjects)

        return train_x,train_y

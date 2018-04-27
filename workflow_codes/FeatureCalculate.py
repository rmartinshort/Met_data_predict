#!/usr/bin/env python
#RMS 2018
#Class that calculates time series features that might be useful in the DAS project
import numpy as np
import scipy.stats
from scipy import signal, fft

class FeatureCalc:

    '''Class that calculates a generic set of timeseries features that can be used for a machine learning problem.
    It is initialized on a timeseries object, which can then be reset with each new timeseries. This might be inefficient so we should
    test it. For each ts array, an array of features is calculated and returned'''

    #String names of all the features. This is useful when building a pandas df from them
    #This class variable should be added to if more features are introduced
    feature_names = ['abs_energy','cid_ce','mean_abs_change','mean_change','mean','median','skewness','kurtosis','interquartile_range',
            'variance','x_crossing_m','maximum','minimum','root_mean_square','fft_energy','peak_welch']


    def __init__(self,time_vector=None,user_feature_list=False):

        '''ts is a timseries of values. time_vector (not currently used) is a vector of times corresponding to those values
        These input timeseries must be of the same length'''


        self.ts = None
        self.samp = int(40) #default sampling rate for BB data
        self.feature_list = user_feature_list

        #Some features may need to import a time vector too, but we won't do this at the moment

        #if len(self.ts) != len(self.time_vector):
        #    raise ValueError("Input timeseries and time vectors are not of the same length!")
        #    sys.exit(1)

        #total number of features that we can calculate
        self.Nfeatures = len(self.feature_names)

        if self.feature_list == False:

            #Assume we want to calculate all features, otherwise just calculate a
            #subset of them

            self.feature_array = np.zeros(self.Nfeatures)

        else:

            #We should be able to use this to specify a subset of features to calculate, in case we don't
            #want all of them

            self.feature_array = np.zeros(len(self.feature_list))

    def return_feature_names(self):

        '''Return the list of feature names
        '''

        return self.feature_names

    def calculate_all_features(self):

        '''Fill the empty feature array with calculated features and return
        '''

        features = [self.abs_energy,self.cid_ce,self.mean_abs_change,self.mean_change,self.meanval,self.medianval,
        self.skewness,self.kurtosis,self.interquartile_range,
        self.variance,self.x_crossing_m,self.maximum,self.minimum,self.root_mean_square,self.fft_energy,
        self.peak_welch]

        for i in range(len(features)):
            self.feature_array[i] = features[i]()

        return self.feature_array

    def load_new_ts(self,input_array):

        '''Updates the class's timeseries object
        '''

        self.ts = np.array(input_array)

    def load_sample_rate(self,input_rate):

        '''Update the class's sample rate. This should only need to be done once, but
        in the case of timeseries with different sampling rates it can be added if needed'''

        self.samp = int(input_rate)

    ##########################################################################################
    # FEATURE CALCULATORS
    ##########################################################################################

    #############################################################
    #  TIME FEATURES

    def abs_energy(self):

        '''Retuns the absolute energy of the ts, which is the sum of its squared values
        tsfresh feature
        F1
        '''

        return sum(self.ts*self.ts)

    def cid_ce(self):

        '''Returns an estimate of time series complexity
        tsfresh feature
        F2
        '''

        x = np.diff(self.ts)
        return np.sqrt(np.sum(x*x))

    def mean_abs_change(self):

        '''Returns mean over absolute differences between subsequent time series values
        tsfresh feature
        F3
        '''

        return np.mean(abs(np.diff(self.ts)))

    def mean_change(self):

        '''Retuns the mean over the differences between subsequent time series values
        tsfresh feature
        F4
        '''

        return np.mean(np.diff(self.ts))

    def meanval(self):

        '''Returns the mean of the timeseries
        Qingkai feature
        F5
        '''
        #Set the mean to a public variable so that other functions can access it if needed
        self.mean = np.mean(self.ts)

        return self.mean

    def medianval(self):

        '''Returns the median of the timeseries
        Qingkai feature
        F6
        '''

        return np.median(self.ts)

    def skewness(self):

        '''Returns the skewness of the timeseries (measure of departure from normal dist)
        Qingkai feature
        F7
        '''

        return scipy.stats.skew(self.ts)

    def kurtosis(self):

        '''Return the kurtosis (4th central moment / square of variance) of the
        timeseries
        Qingkai feature
        F8
        '''

        return scipy.stats.kurtosis(self.ts)

    def interquartile_range(self):

        '''Return the iqr of the timeseries
        Qingkai feature
        F9
        '''

        return scipy.stats.iqr(self.ts)

    def variance(self):

        '''Return the variance of the timeseries
        Qingkai feature
        F10
        '''

        return np.var(self.ts)


    def maximum(self):

        '''Return the maximum value in the timseries
        tsfresh feature
        F12
        '''

        return max(self.ts)

    def minimum(self):

        '''Return the minimum value in the timeseries
        tsfresh feature
        F13
        '''

        return min(self.ts)

    def root_mean_square(self):

        '''Return the root_mean_square of the timeseries
        Qingkai feature
        F14
        '''
        x = self.ts
        return np.sqrt(np.mean(x*x))

    #############################################################
    #  FREQUENCY FEATURES


    def fft_energy(self):

        '''Return the sum of the amplitude spectrum
        Qingkai feature
        F15
        '''
        n = int(len(self.ts)/2)
        fft_amp = np.abs(fft(self.ts)[np.arange(1,n)])
        return np.sum(fft_amp**2)/len(fft_amp) 

    def peak_welch(self):

        '''Return the frequency corresponding to the maximum value in the 
        Welch peridogram
        RMS feature
        F16
        '''

        f, pxx = signal.welch(self.ts,self.samp,window='hanning',nperseg=64)
        return f[np.argmax(pxx)]

    def x_crossing_m(self):

        '''Return the number of m crossings in the timeseries. Typically m is the mean
        tsfresh feature
        F11
        '''

        x = self.ts[self.ts != self.mean]
        return sum(np.abs(np.diff(np.sign(x-self.mean))))/2.0



if __name__ == "__main__":

    a = FeatureCalc()
    a.load_new_ts([1,2,3,4])

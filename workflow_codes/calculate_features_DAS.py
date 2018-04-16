import numpy as np
import scipy.stats
import scipy.integrate as sp_int
from scipy import fft, arange
import pywt

'''
This is the script to calculate the features from the acceleration data sent by the
phone, there are two functions, one is to calculate the features for the training,
this one doesn't need in real time, and the other one (features_rt function) is to
calculate the features in real time.
'''

def calculate_z(a, nsta):
    """
    Z-detector.

    :param nsta: Window length in Samples.

    .. seealso:: [Withers1998]_, p. 99
    """
    m = len(a)
    #
    # Z-detector given by Swindell and Snell (1977)
    sta = np.zeros(len(a), dtype='float64')
    # Standard Sta
    pad_sta = np.zeros(nsta)
    for i in xrange(nsta):  # window size to smooth over
        sta = sta + np.concatenate((pad_sta, a[i:m - nsta + i] ** 2))
    a_mean = np.mean(sta)
    a_std = np.std(sta)
    Z = (sta - a_mean) / a_std


def mean_abs_dev(pop):
    n = float(len(pop))
    mean = sum(pop) / n
    diff = [ abs(x - mean) for x in pop ]
    return np.sqrt(sum(diff) / (n - 1))

def rms(num):
    return np.sqrt(sum(n*n for n in num))/len(num)

def features_rt(time_, X):
    '''
    function to calculate features for a single DAS trace. X is the trace vector for
    time of length time__

    Features as follows
    Feature1 - mean
    Feature2 - median
    Feature3 - 25% percentile, 75% Percentile, interquartile range
    Feature4 - PGA, PGV, PGD
    Feature5 - minmax (range),
    Feature6 - taoVA
    Feature7 - Variance
    Feature8 - STD, mean absolute deviation
    Feature9 - RMS
    Feature10 - skewness
    Feature11 - kurtosis
    Feature12 - Zero crossings Rate
    Feature13 - CAV
    Feature14 - Z score for the peak value
    Feature15 - correlation between each axis
    Feature16 - coeff. of A and B (optional)


    energy, entropy, centroid frequency, peak frequency, FFT, Discrete Cosine Transform

    Wavelet coefficients, Energy distribution, RMS velocity FFT
    '''

    #calculate vector sum of input trace
    vector_sum = np.sqrt(np.square(X))

    #Feature0
    #SMA - Signal Magnitude Area
    sma_1_comp = sp_int.simps(y = np.abs(X), x = time_)

    #Feature1
    mean_vector_sum = np.mean(vector_sum)

    #Feature2
    median_vector_sum = np.median(vector_sum)

    #Feature3
    q25_vector_sum = np.percentile(vector_sum, q = 25)
    q75_vector_sum = np.percentile(vector_sum, q = 75)
    iqr_vector_sum = q75_vector_sum - q25_vector_sum

    #Feature4
    max_vector_sum = np.max(vector_sum)

    #Feature5
    minmax_vector_sum =np.ptp(vector_sum)

    #Feature6
    var_vector_sum = np.var(vector_sum)

    #Feature8
    std_vector_sum = np.std(vector_sum)
    mad_vector_sum = mean_abs_dev(vector_sum)

    #Feature9
    rms_vector_sum = rms(vector_sum)

    #Feature10
    skew_vector_sum = scipy.stats.skew(vector_sum)

    #Feature11
    kurtosis_vector_sum = scipy.stats.kurtosis(vector_sum)
    k2_vector_sum = np.power(skew_vector_sum,2) + np.power(kurtosis_vector_sum,2)

    #Feature12
    #calculate the zero crossing rate for each of the component
    X_crossing = np.where(np.diff(np.sign(X)))[0]
    xc_max = len(X_crossing) #the number of zero crossings

    #Feature13
    cav_vector_sum = sp_int.simps(y = np.abs(vector_sum), x = time_)

    #Feature14
    Z_vector_sum = (np.max(np.abs(vector_sum)) - mean_vector_sum) / var_vector_sum

    #Feature15
    n = len(X) # length of the signal
    #k = arange(n)
    #T = int(n/25)
    #frq = k/T # two sides frequency range
    #frq = frq[int(range(n/2))] # one side frequency range

    fft_X = fft(X) # fft computing
    fft_vector_sum = fft(vector_sum)

    fft_amp_X = np.abs(fft_X[arange(1, n/2).astype(int)])
    fft_amp_vector_sum = np.abs(fft_vector_sum[arange(1, n/2).astype(int)])

    energy_x = np.sum(fft_amp_X**2) / len(fft_amp_X)

    #energy_vector_sum = np.sum(fft_vector_sum**2) / len(fft_vector_sum)

    # #Feature16
    # #Features from the wavelets decomposition
    # #for sampling rate 25 Hz,
    # #A1 ~ 0 - 0.39 Hz
    # #D1 ~ 0.39 - 0.78 Hz
    # #D2 ~ 0.78 - 1.56 Hz
    # #D3 ~ 1.56 - 3.12 Hz
    # #D4 ~ 3.12 - 6.25 Hz
    # #D5 ~ 6.25 - 12.5 Hz
    # #D6 ~ 12.5 - 25 Hz (ignored)
    # A1_vector_sum, D1_vector_sum, D2_vector_sum, D3_vector_sum, D4_vector_sum, D5_vector_sum, \
    #     D6_vector_sum = pywt.wavedec(vector_sum, 'db1', level=6)
    #
    # #we ignore the last coeff., since it corresponding to 12.5 - 25 Hz, which our sampling rate is
    # #only 25 Hz
    #
    # #Feature18 - Feature23
    # #47
    # powerA1_vector_sum = np.sum(A1_vector_sum**2) / len(A1_vector_sum)
    # #48
    # powerD1_vector_sum = np.sum(D1_vector_sum**2) / len(D1_vector_sum)
    # #49
    # powerD2_vector_sum = np.sum(D2_vector_sum**2) / len(D2_vector_sum)
    # #50
    # powerD3_vector_sum = np.sum(D3_vector_sum**2) / len(D3_vector_sum)
    # #51
    # powerD4_vector_sum = np.sum(D4_vector_sum**2) / len(D4_vector_sum)
    # #52
    # powerD5_vector_sum = np.sum(D5_vector_sum**2) / len(D5_vector_sum)
    #
    # A1_x, D1_x, D2_x, D3_x, D4_x, D5_x, D6_x = pywt.wavedec(X, 'db1', level=6)
    # A1_y, D1_y, D2_y, D3_y, D4_y, D5_y, D6_y = pywt.wavedec(Y, 'db1', level=6)
    # A1_z, D1_z, D2_z, D3_z, D4_z, D5_z, D6_z = pywt.wavedec(Z, 'db1', level=6)
    #
    # #Feature24 - Feature29
    # #53
    # powerA1_all = (np.sum(A1_x**2) / len(A1_x) + np.sum(A1_y**2) / len(A1_y) + np.sum(A1_z**2) / len(A1_z))/3
    # #54
    # powerD1_all = (np.sum(D1_x**2) / len(D1_x) + np.sum(D1_y**2) / len(D1_y) + np.sum(D1_z**2) / len(D1_z))/3
    # #55
    # powerD2_all = (np.sum(D2_x**2) / len(D2_x) + np.sum(D2_y**2) / len(D2_y) + np.sum(D2_z**2) / len(D2_z))/3
    # #56
    # powerD3_all = (np.sum(D3_x**2) / len(D3_x) + np.sum(D3_y**2) / len(D3_y) + np.sum(D3_z**2) / len(D3_z))/3
    # #57
    # powerD4_all = (np.sum(D4_x**2) / len(D4_x) + np.sum(D4_y**2) / len(D4_y) + np.sum(D4_z**2) / len(D4_z))/3
    # #58
    # powerD5_all = (np.sum(D5_x**2) / len(D5_x) + np.sum(D5_y**2) / len(D5_y) + np.sum(D5_z**2) / len(D5_z))/3
    #print(sma_1_comp,mean_vector_sum,median_vector_sum,iqr_vector_sum,minmax_vector_sum,var_vector_sum,std_vector_sum,mad_vector_sum,rms_vector_sum,skew_vector_sum,kurtosis_vector_sum,k2_vector_sum,xc_max,cav_vector_sum,Z_vector_sum,energy_x)

    return sma_1_comp, mean_vector_sum, median_vector_sum, iqr_vector_sum,\
    minmax_vector_sum, var_vector_sum, std_vector_sum, mad_vector_sum, rms_vector_sum,\
    skew_vector_sum, kurtosis_vector_sum, k2_vector_sum, xc_max, cav_vector_sum, Z_vector_sum, energy_x

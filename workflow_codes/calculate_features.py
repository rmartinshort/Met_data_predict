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

def features_rt(time_, X, Y, Z, velX, velY, velZ, dispX, dispY, dispZ):
    '''
    function to calculate features for in real time, so X, Y and Z are the time windows
    in real time (e.g. 2 seconds and so on). vel and disp are the velocity and displacement
    
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
    
    #calculate vector sum of the 3 ACC amplitudes
    acc_vector_sum = np.sqrt(np.square(X)+np.square(Y)+np.square(Z))
    vel_vector_sum = np.sqrt(np.square(velX)+np.square(velY)+np.square(velZ))
    disp_vector_sum = np.sqrt(np.square(dispX)+np.square(dispY)+np.square(dispZ))
    
    #Feature0
    #0
    #SMA - Signal Magnitude Area
    sma_3_comp = sp_int.simps(y = np.abs(X), x = time_) + sp_int.simps(y = np.abs(Y), x = time_) + \
        sp_int.simps(y = np.abs(Z), x = time_)
        
    #Feature1
    #1
    mean_acc_vector_sum = np.mean(acc_vector_sum)
    #2
    mean_vel_vector_sum = np.mean(vel_vector_sum)
    #3
    mean_disp_vector_sum = np.mean(disp_vector_sum)
    
    #Feature2
    #4
    median_acc_vector_sum = np.median(acc_vector_sum)
    #5
    median_vel_vector_sum = np.median(vel_vector_sum)
    #6
    median_disp_vector_sum = np.median(disp_vector_sum)
    
    #Feature3
    q25_acc_vector_sum = np.percentile(acc_vector_sum, q = 25)
    q25_vel_vector_sum = np.percentile(vel_vector_sum, q = 25)
    q25_disp_vector_sum = np.percentile(disp_vector_sum, q = 25)
    
    q75_acc_vector_sum = np.percentile(acc_vector_sum, q = 75)
    q75_vel_vector_sum = np.percentile(vel_vector_sum, q = 75)
    q75_disp_vector_sum = np.percentile(disp_vector_sum, q = 75)
    
    #7
    iqr_acc_vector_sum = q75_acc_vector_sum - q25_acc_vector_sum
    #8
    iqr_vel_vector_sum = q75_vel_vector_sum - q25_vel_vector_sum
    #9
    iqr_disp_vector_sum = q75_disp_vector_sum - q25_disp_vector_sum
    
    #Feature4
    #10
    pga_vector_sum = np.max(acc_vector_sum)
    #11
    pgv_vector_sum = np.max(vel_vector_sum)
    #12
    pgd_vector_sum = np.max(disp_vector_sum)
    
    #Feature5
    #13
    minmax_acc_vector_sum =np.ptp(acc_vector_sum)
    #14
    minmax_vel_vector_sum = np.ptp(vel_vector_sum)
    #15
    minmax_disp_vector_sum = np.ptp(disp_vector_sum)
    
    #Feature6
    #taoVA is a measure of the frequency
    #16
    taoVA = (np.max(vel_vector_sum) / np.max(acc_vector_sum)) * 2. * np.pi
    
    #Feature7
    #17
    var_acc_vector_sum = np.var(acc_vector_sum)
    #18
    var_vel_vector_sum = np.var(vel_vector_sum)
    #19
    var_disp_vector_sum = np.var(disp_vector_sum)
    
    #Feature8
    #20
    std_acc_vector_sum = np.std(acc_vector_sum)
    #21
    std_vel_vector_sum = np.std(vel_vector_sum)
    #22
    std_disp_vector_sum = np.std(disp_vector_sum)
    
    #23
    mad_acc_vector_sum = mean_abs_dev(acc_vector_sum)
    #24
    mad_vel_vector_sum = mean_abs_dev(vel_vector_sum)
    #25
    mad_disp_vector_sum = mean_abs_dev(disp_vector_sum)
    
    #Feature9
    #26
    rms_acc_vector_sum = rms(acc_vector_sum)
    #27
    rms_vel_vector_sum = rms(vel_vector_sum)
    #28
    rms_disp_vector_sum = rms(disp_vector_sum)
    
    #Feature10
    #29
    skew_acc_vector_sum = scipy.stats.skew(acc_vector_sum)
    #30
    skew_vel_vector_sum = scipy.stats.skew(vel_vector_sum)
    #31
    skew_disp_vector_sum = scipy.stats.skew(disp_vector_sum)
    
    #Feature11
    #32
    kurtosis_acc_vector_sum = scipy.stats.kurtosis(acc_vector_sum)
    #33
    kurtosis_vel_vector_sum = scipy.stats.kurtosis(vel_vector_sum)
    #34
    kurtosis_disp_vector_sum = scipy.stats.kurtosis(disp_vector_sum)
    #35
    k2_acc_vector_sum = np.power(skew_acc_vector_sum,2) + np.power(kurtosis_acc_vector_sum,2)
    
    #Feature12
    #calculate the zero crossing rate for each of the component
    X_crossing = np.where(np.diff(np.sign(X)))[0]
    Y_crossing = np.where(np.diff(np.sign(Y)))[0]
    Z_crossing = np.where(np.diff(np.sign(Z)))[0]
    
    #maximum zero crossing rate
    #36
    zc_max = max(len(X_crossing), len(Y_crossing), len(Z_crossing))
    
    #Feature13
    #37
    cav_acc_vector_sum = sp_int.simps(y = np.abs(acc_vector_sum), x = time_)
    
    #Feature14
    #38
    Z_acc_vector_sum = (np.max(np.abs(acc_vector_sum)) - mean_acc_vector_sum) / var_acc_vector_sum
    
    #Feature15
    #39
    corr_xy = scipy.stats.pearsonr(X, Y)[0]
    #40
    corr_xz = scipy.stats.pearsonr(X, Z)[0]
    #41
    corr_yz = scipy.stats.pearsonr(Y, Z)[0]
    
    #Feature16
    n = len(X) # length of the signal
    k = arange(n)
    T = n/25.
    frq = k/T # two sides frequency range
    frq = frq[range(n/2)] # one side frequency range

    fft_X = fft(X) # fft computing
    fft_Y = fft(Y)
    fft_Z = fft(Z)
    fft_vector_sum = fft(acc_vector_sum)
    
    fft_amp_X = np.abs(fft_X[arange(1, n/2)])
    fft_amp_Y = np.abs(fft_Y[arange(1, n/2)])
    fft_amp_Z = np.abs(fft_Z[arange(1, n/2)])
    fft_amp_vector_sum = np.abs(fft_vector_sum[range(1, n/2)])
    
    #42
    energy_x = np.sum(fft_amp_X**2) / len(fft_amp_X)
    #43
    energy_y = np.sum(fft_amp_Y**2) / len(fft_amp_Y)
    #44
    energy_z = np.sum(fft_amp_Z**2) / len(fft_amp_Z)
    #45
    energy_al_ = energy_x + energy_y + energy_z / 3
f_ticks
    #46
    energy_acc_vector_sum = np.sum(fft_amp_vector_sum**2) / len(fft_amp_vector_sum)
    
    #Feature17
    #Features from the wavelets decomposition 
    #for sampling rate 25 Hz, 
    #A1 ~ 0 - 0.39 Hz
    #D1 ~ 0.39 - 0.78 Hz
    #D2 ~ 0.78 - 1.56 Hz
    #D3 ~ 1.56 - 3.12 Hz
    #D4 ~ 3.12 - 6.25 Hz
    #D5 ~ 6.25 - 12.5 Hz
    #D6 ~ 12.5 - 25 Hz (ignored)
    A1_vector_sum, D1_vector_sum, D2_vector_sum, D3_vector_sum, D4_vector_sum, D5_vector_sum, \
        D6_vector_sum = pywt.wavedec(acc_vector_sum, 'db1', level=6)
    
    #we ignore the last coeff., since it corresponding to 12.5 - 25 Hz, which our sampling rate is 
    #only 25 Hz
    
    #Feature18 - Feature23
    #47
    powerA1_vector_sum = np.sum(A1_vector_sum**2) / len(A1_vector_sum)
    #48
    powerD1_vector_sum = np.sum(D1_vector_sum**2) / len(D1_vector_sum)
    #49
    powerD2_vector_sum = np.sum(D2_vector_sum**2) / len(D2_vector_sum)
    #50
    powerD3_vector_sum = np.sum(D3_vector_sum**2) / len(D3_vector_sum)
    #51
    powerD4_vector_sum = np.sum(D4_vector_sum**2) / len(D4_vector_sum)
    #52
    powerD5_vector_sum = np.sum(D5_vector_sum**2) / len(D5_vector_sum)
    
    A1_x, D1_x, D2_x, D3_x, D4_x, D5_x, D6_x = pywt.wavedec(X, 'db1', level=6)
    A1_y, D1_y, D2_y, D3_y, D4_y, D5_y, D6_y = pywt.wavedec(Y, 'db1', level=6)
    A1_z, D1_z, D2_z, D3_z, D4_z, D5_z, D6_z = pywt.wavedec(Z, 'db1', level=6)
    
    #Feature24 - Feature29
    #53
    powerA1_all = (np.sum(A1_x**2) / len(A1_x) + np.sum(A1_y**2) / len(A1_y) + np.sum(A1_z**2) / len(A1_z))/3
    #54
    powerD1_all = (np.sum(D1_x**2) / len(D1_x) + np.sum(D1_y**2) / len(D1_y) + np.sum(D1_z**2) / len(D1_z))/3
    #55
    powerD2_all = (np.sum(D2_x**2) / len(D2_x) + np.sum(D2_y**2) / len(D2_y) + np.sum(D2_z**2) / len(D2_z))/3
    #56
    powerD3_all = (np.sum(D3_x**2) / len(D3_x) + np.sum(D3_y**2) / len(D3_y) + np.sum(D3_z**2) / len(D3_z))/3
    #57
    powerD4_all = (np.sum(D4_x**2) / len(D4_x) + np.sum(D4_y**2) / len(D4_y) + np.sum(D4_z**2) / len(D4_z))/3
    #58
    powerD5_all = (np.sum(D5_x**2) / len(D5_x) + np.sum(D5_y**2) / len(D5_y) + np.sum(D5_z**2) / len(D5_z))/3
    
    return [sma_3_comp, mean_acc_vector_sum, mean_vel_vector_sum, mean_disp_vector_sum, median_acc_vector_sum, \
    median_vel_vector_sum, median_disp_vector_sum, iqr_acc_vector_sum, iqr_vel_vector_sum, \
    iqr_disp_vector_sum, pga_vector_sum, pgv_vector_sum, pgd_vector_sum, minmax_acc_vector_sum, \
    minmax_vel_vector_sum, minmax_disp_vector_sum, taoVA, var_acc_vector_sum, var_vel_vector_sum, \
    var_disp_vector_sum, std_acc_vector_sum, std_vel_vector_sum, std_disp_vector_sum, \
    mad_acc_vector_sum, mad_vel_vector_sum, mad_disp_vector_sum, rms_acc_vector_sum, rms_vel_vector_sum, \
    rms_disp_vector_sum, skew_acc_vector_sum, skew_vel_vector_sum, skew_disp_vector_sum, \
    kurtosis_acc_vector_sum, kurtosis_vel_vector_sum, kurtosis_disp_vector_sum, k2_acc_vector_sum, \
    zc_max, cav_acc_vector_sum, Z_acc_vector_sum, corr_xy, corr_xz, corr_yz, energy_x, energy_y, energy_z, \
    energy_all, energy_acc_vector_sum, powerA1_vector_sum, powerD1_vector_sum, powerD2_vector_sum, \
    powerD3_vector_sum, powerD4_vector_sum, powerD5_vector_sum, powerA1_all, powerD1_all, powerD2_all, \
    powerD3_all, powerD4_all, powerD5_all]


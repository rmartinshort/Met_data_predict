#!/usr/bin/env python
#RMS 2018
#Functions to aid in DAD dataset processing
import numpy as np

#note that the following import only works with latest obspy
from obspy.signal.util import nearest_pow_2 as nextpow2

from scipy.fftpack import fft, ifft, fftshift, ifftshift


def despike(trace,scale_factor=200):

    '''Remove all data whose amplitude is larger than scale factor x median of the non-nan parts of trace
    This follows the method of Bakku, 2010'''

    data = trace.data
    #determine the value above which a spike is declared
    cutoff = scale_factor*np.nanmedian(abs(data))
    #This will produce a warning, but the nan values will remain nan
    data[abs(data) > cutoff] = np.nan
    trace.data = data

    return trace


def despike_zero(trace,scale_factor=200):

    '''Remove all data whose amplitude is larger than scale factor x median of the non-nan parts of trace
    This follows the method of Bakku, 2010'''

    data = trace.data
    #determine the value above which a spike is declared
    cutoff = scale_factor*np.nanmedian(abs(data))
    #This will produce a warning, but the nan values will remain nan
    data[abs(data) > cutoff] = 0.0
    trace.data = data

    return trace


def remove_median(trace):

    '''Remove median from a trace that contains NaNs'''

    #remove the median of the trace
    data = trace.data
    median_non_nan = np.nanmedian(data)
    data = data - median_non_nan
    trace.data = data

    return trace

def despike_np_array(array,scale_factor=200):

    '''Remove all data whose amplitude is larger than scale factor x median of the non-nan parts of trace
    This follows the method of Bakku, 2010

    This works for np arrays only'''

    data = array
    #determine the value above which a spike is declared
    cutoff = scale_factor*np.nanmedian(abs(data))
    #This will produce a warning, but the nan values will remain nan
    data[abs(data) > cutoff] = np.nan

    return data


def fillArray(data, mask=None, fill_value=None):
    """
    from trichter/sito/utils/helper.py

    Fill masked numpy array with value without demasking.

    Additonally set fill_value to value.
    If data is not a MaskedArray returns silently data.
    """
    if mask is not None and mask is not False:
        data = np.ma.MaskedArray(data, mask=mask, copy=False)
    if np.ma.is_masked(data) and fill_value is not None:
        data._data[data.mask] = fill_value
        np.ma.set_fill_value(data, fill_value)
    elif not np.ma.is_masked(data):
        data = np.ma.filled(data)
    return data

def spectralWhitening(data, sr=None, smoothi=None, freq_domain=False):
    """
    From trichter/sito/xcorr.py

    Apply spectral whitening to data.

    sr: sampling rate (only needed for smoothing)
    smoothi: None or int
    Data is divided by its smoothed (Default: None) amplitude spectrum.
    """
    if freq_domain:
        mask = False
        spec = data
    else:
        mask = np.ma.getmask(data)
        N = len(data)
        nfft = nextpow2(N)
        spec = fft(data, nfft)

    spec_ampl = np.sqrt(np.abs(np.multiply(spec, np.conjugate(spec))))
    if smoothi:
        smoothi = int(smoothi * N / sr)
        spec /= ifftshift(smooth(fftshift(spec_ampl), smoothi))
    else:
        spec /= spec_ampl

    if freq_domain:
        return spec
    else:
        ret = np.real(ifft(spec, nfft)[:N])

    return fillArray(ret,mask=mask,fill_value=0)

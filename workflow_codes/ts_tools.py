#!/usr/bin/env python
#RMS 2018
#Functions to aid in DAD dataset processing
import numpy as np

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

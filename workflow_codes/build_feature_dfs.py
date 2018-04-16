#!/usr/bin/env python
#RMS 2018
#Build dataframes of features for each of the mseed trace files

import pandas as pd
import numpy as np
import obspy as op
import glob

#custom modules
from calculate_features_DAS import *
from ts_tools import despike, remove_median


def main():

    streams = glob.glob('channel*.mseed')
    time = 2 #length of time, in seconds, to calculate features over


    for instream in streams:

        print("Dealing with stream %s" %instream)

        #name of dataframe .csv file to write
        outdfname = '%s_features_%is_df.csv' %(instream[:-6],int(time))
        print("Outfile name will be %s" %outdfname)

        stream = op.read(instream,format='mseed')
        tinc = stream[0].stats.delta
        window = int(time/tinc) #number of samples per unit time
        npts = int(stream[0].stats.npts)

        stream.merge(method=1,fill_value=np.NaN)

        #remove median
        trace = remove_median(stream[0])
        #run despike
        trace = despike(trace)
        stream[0] = trace

        #length of each of the feature vectors
        ltrace = npts-window

        #feature vectors. fill with nan initially
        sma_1_comps = np.full(ltrace,np.nan)
        means = np.full(ltrace,np.nan)
        medians = np.full(ltrace,np.nan)
        iqrs = np.full(ltrace,np.nan)
        minmaxs = np.full(ltrace,np.nan)
        variances = np.full(ltrace,np.nan)
        stds = np.full(ltrace,np.nan)
        mads = np.full(ltrace,np.nan)
        rmss = np.full(ltrace,np.nan)
        skews = np.full(ltrace,np.nan)
        kurtosiss = np.full(ltrace,np.nan)
        k2s = np.full(ltrace,np.nan)
        Xcrossings = np.full(ltrace,np.nan)
        cavs = np.full(ltrace,np.nan)
        Zs = np.full(ltrace,np.nan)
        energys = np.full(ltrace,np.nan)
        time_vector = np.full(ltrace,np.nan)

        #slide the window and extract data
        trace = stream[0].data
        time__ = np.linspace(0,time,window)

        #Loop to determine the features from each 2 second slice of the trace
        print("Entering feature calculation loop")
        for i in range(ltrace):
            inslice = trace[i:i+window]
            time_vector[i] = i*tinc + time/2 #time corresponding to the middle of the trace
            if not np.isnan(inslice).any():
                    sma_1_comps[i], means[i], medians[i], iqrs[i],\
                minmaxs[i], variances[i], stds[i], mads[i], rmss[i],\
                skews[i], kurtosiss[i], k2s[i], Xcrossings[i], cavs[i],\
                Zs[i], energys[i] = features_rt(time__,inslice)

        #Gather the feature vectors into a dataframe and write
        df = pd.DataFrame(data={'time':time_vector,'sma_1':sma_1_comps,'mean':means,'medians':medians,
        'iqr':iqrs,'minmax':minmaxs,'variance':variances,'std':stds,'mads':mads,'rms':rmss,'skew':skews,
        'kurtosis':kurtosiss,'k2s':k2s,'xcs':Xcrossings,'cav':cavs,'Z':Zs,'energy':energys})
        df.to_csv(outdfname,index=False)

if __name__ == "__main__":

    main()

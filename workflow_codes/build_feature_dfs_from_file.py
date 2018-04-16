#!/usr/bin/env python
#RMS 2018
#Alternative method for constucting features. In this case, we calculate one set of features per input DAS file, and
#generate dataframes for each of the features across all channels. These can then be directly linked to the soil
#moisture dataframe

import pandas as pd
import obspy as op
import numpy as np
from calculate_features_DAS import *
from ts_tools import despike, remove_median
import os

def main():

    all_data = pd.read_csv("Fnames_soilM_interp.csv")
    lineCSN = all_data[all_data['Line'] == "LineCSN"]

    #Extract just the soil moisture time series corresponding to probe M3_20
    lineCSN_M3 = lineCSN[['file_name','DateTime','M3_20_Avg_interp','tsince_start']]
    lineCSN_M3.reset_index(drop=True,inplace=True)
    lineCSN_M3['DateTime'] = pd.to_datetime(lineCSN_M3['DateTime'])
    lineCSN_M3.set_index('DateTime',inplace=True)
    fnames = list(lineCSN['file_name'])

    #numbered list of the trace indices to extract
    traces = range(75,95)

    #Location of the DAS data files
    datadir = "/media/rmartinshort/My Book/4Robert"
    cwd = os.getcwd()
    os.chdir(datadir)

    #Set up feature matrices
    ltrace = len(fnames) #number of traces to loop over
    nchannels = len(traces) #number of channels to extract
    sma_1_comps = np.full([ltrace,nchannels],np.nan)
    means = np.full([ltrace,nchannels],np.nan)
    medians = np.full([ltrace,nchannels],np.nan)
    iqrs = np.full([ltrace,nchannels],np.nan)
    minmaxs = np.full([ltrace,nchannels],np.nan)
    variances = np.full([ltrace,nchannels],np.nan)
    stds = np.full([ltrace,nchannels],np.nan)
    mads = np.full([ltrace,nchannels],np.nan)
    rmss = np.full([ltrace,nchannels],np.nan)
    skews = np.full([ltrace,nchannels],np.nan)
    kurtosiss = np.full([ltrace,nchannels],np.nan)
    k2s = np.full([ltrace,nchannels],np.nan)
    Xcrossings = np.full([ltrace,nchannels],np.nan)
    cavs = np.full([ltrace,nchannels],np.nan)
    Zs = np.full([ltrace,nchannels],np.nan)
    energys = np.full([ltrace,nchannels],np.nan)

    feature_names = ['sma_1_comp','mean','median','iqr','minmax',\
    'variance','std','mad','rms','skew','kurtosis','k2','xcrossing_rate','cav','Z','energy']

    i = 0
    file_name_list = []
    for fname in fnames:

        print ('\nCalculating features for file %s\n' %fname)

        #Save the name of the file
        file_name_list.append(fname[2:])

        #Read in file
        f = op.read(fname,format='mseed')

        #LOOP OVER THE CHANNELS WE WANT TO EXTRACT, SELECT THE CHANNEL, CALCULATE FEATURES AND
        #APPEND TO FEATURE MATRIX

        j = 0
        for traceid in traces:

            trace = f[traceid]
            trace = remove_median(trace)
            trace = despike(trace)
            st = op.UTCDateTime(trace.stats.starttime)
            et = op.UTCDateTime(trace.stats.endtime)

            time__ = np.linspace(0,(et-st),trace.stats.npts)
            inslice = trace.data

            if not np.isnan(inslice).any():

                sma_1_comps[i,j], means[i,j], medians[i,j], iqrs[i,j],\
                    minmaxs[i,j], variances[i,j], stds[i,j], mads[i,j], rmss[i,j],\
                    skews[i,j], kurtosiss[i,j], k2s[i,j], Xcrossings[i,j], cavs[i,j],\
                    Zs[i,j], energys[i,j] = features_rt(time__,inslice)
            else:
                print ('nan found in %s' %fname)

            j += 1
        i += 1

    #Create names vector for the columns
    channel_names = []
    for traceid in traces:
        name = 'channel_%i' %traceid
        channel_names.append(name)

    #Write each matrix to a pd dataframe and to outfile
    i = 0
    for matrix in [sma_1_comps,means,medians,iqrs,minmaxs,variances,stds,mads,rmss,skews,kurtosiss,k2s,Xcrossings,cavs,Zs,energys]:
        df = pd.DataFrame(matrix,columns=channel_names)
        df['file_name'] = file_name_list
        df.to_csv(feature_names[i]+'.csv',index=False)
        i += 1

    os.system('mv *.csv %s' %cwd)
    os.chdir(cwd)

if __name__ == "__main__":
    main()

#!/usr/bin/env python
#RMS 2018
#Alternative method for constucting features. In this case, we calculate one set of features per input DAS file, and
#generate dataframes for each of the features across all channels. These can then be directly linked to the soil
#moisture dataframe

import pandas as pd
import obspy as op
import numpy as np
from ts_tools import despike
import FeatureCalculate_new as FC
import os

def main():

    all_data = pd.read_csv("Fnames_soilM_temp.csv")
    lineCSN = all_data[all_data['Line'] == "LineCSN"]

    #Extract just the soil moisture time series corresponding to probe M3_20, and the regional temperature
    #These will be the variables we want to train on

    lineCSN_M3 = lineCSN[['file_name','DateTime','M3_20_Avg_interp','Temphigh_interp']]
    lineCSN_M3.reset_index(drop=True,inplace=True)
    #Note that to avoid warning we should use assign to replace column values
    lineCSN_M3.assign(DateTime = pd.to_datetime(lineCSN_M3['DateTime']).values)
    otime = op.UTCDateTime(lineCSN_M3['DateTime'][0])
    lineCSN_M3.set_index('DateTime',inplace=True)

    #Generate linking column
    lineCSN_M3 = lineCSN_M3.assign(file_name=lineCSN_M3['file_name'].apply(lambda x: x[2:]))
    fnames = lineCSN['file_name']

    #numbered list of the trace indices to extract
    t1 = 80
    t2 = 90
    traces = range(t1,t2)

    #Location of the DAS data files
    datadir = "/media/rmartinshort/My Book/4Robert"
    cwd = os.getcwd()
    os.chdir(datadir)

    #rolling window charactersitics in terms of number of files
    rolling_window_len = 30
    rolling_window_offset = 5
    len_trace_seconds = rolling_window_len*60.0

    #Set up feature matrices
    ltrace = divmod(len(fnames),rolling_window_offset)[0] - 1
    nchannels = len(traces) #number of channels to extract

    #Preallocate feature matrices
    f1 = np.full([ltrace,nchannels],np.nan)
    f2 = np.full([ltrace,nchannels],np.nan)
    f3 = np.full([ltrace,nchannels],np.nan)
    f4 = np.full([ltrace,nchannels],np.nan)
    f5 = np.full([ltrace,nchannels],np.nan)
    f6 = np.full([ltrace,nchannels],np.nan)
    f7 = np.full([ltrace,nchannels],np.nan)
    f8 = np.full([ltrace,nchannels],np.nan)
    f9 = np.full([ltrace,nchannels],np.nan)
    f10 = np.full([ltrace,nchannels],np.nan)
    f11 = np.full([ltrace,nchannels],np.nan)
    f12 = np.full([ltrace,nchannels],np.nan)
    f13 = np.full([ltrace,nchannels],np.nan)
    f14 = np.full([ltrace,nchannels],np.nan)
    f15 = np.full([ltrace,nchannels],np.nan)
    f16 = np.full([ltrace,nchannels],np.nan)


    FCCalc = FC.FeatureCalc()
    feature_names = FCCalc.return_feature_names()

    #List of timestamps associated with the center time of each trace
    time_markers = []

    for p in range(ltrace):

        start_index = p*rolling_window_offset
        end_index = start_index + rolling_window_len

        print(start_index,end_index)

        chunk = fnames[start_index:end_index]

        print("------------------------------------------------------------")
        print("Dealing with chunk %s" %chunk)


        #Each channel needs to have one stream associated with it.
        #We build these streams by extracting from the mseed files, then
        #we merge them and calculate the features
        channel_streams = []

        for q in range(0,len(traces)):
            channel_streams.append(op.Stream())

        #For file in chunk, read and append traces to the appropriate steam

        for f in chunk.values:
            st = op.read(f,format='mseed')
            i = 0
            for trace in st[t1:t2]:
                channel_streams[i] += trace
                i += 1

        #Loop through the streams we just made, Merge them and calculate features if they do not include NaN values

        for r in range(len(channel_streams)):

            #Merge the traces
            channel_streams[r].merge(method=1,fill_value=np.nan)
            tr = channel_streams[r][0]

            #Apply lowpass filter - filter out all signals above 1 min period. These might not be related to the met dataset
            tr = tr.copy().filter('lowpass',freq=0.01667)

            #Despike filter the traces
            tr = despike(tr,scale_factor=100)

            #Determine the length of the trace that we've just merged. Since there are
            #gaps, some traces will be associated with very large times
            len_trace = op.UTCDateTime(tr.stats.endtime) - op.UTCDateTime(tr.stats.starttime)

            if np.ceil(len_trace) == len_trace_seconds:

                if not np.isnan(tr).any():

                    FCCalc.load_new_ts(tr.data)

                    fa = FCCalc.calculate_all_features()

                    f1[p,r] = fa[0]
                    f2[p,r] = fa[1]
                    f3[p,r] = fa[2]
                    f4[p,r] = fa[3]
                    f5[p,r] = fa[4]
                    f6[p,r] = fa[5]
                    f7[p,r] = fa[6]
                    f8[p,r] = fa[7]
                    f9[p,r] = fa[8]
                    f10[p,r] = fa[9]
                    f11[p,r] = fa[10]
                    f12[p,r] = fa[11]
                    f13[p,r] = fa[12]
                    f14[p,r] = fa[13]
                    f15[p,r] = fa[14]
                    f16[p,r] = fa[15]

        #Time marker corresponds the center of that trace. All traces in the merged stream
        #should be the same length, so we only need to do this calculation on one of them
        #This becomes the time stap at which we interpolate the soil moisture or temperature
        #timeseries
        time_markers.append(op.UTCDateTime(tr.stats.starttime + len_trace/2))

        if p > 0:
            print(time_markers[p] - time_markers[p-1])

    #Create names vector for the columns
    channel_names = []
    for traceid in traces:
        name = 'channel_%i' %traceid
        channel_names.append(name)

    #Write each matrix to a pd dataframe and to outfile
    i = 0
    for matrix in [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16]:
        df = pd.DataFrame(matrix,columns=channel_names)
        df['time'] = time_markers
        df.to_csv(feature_names[i]+'_rms_fcalc_new_sliding_30.csv',index=False)
        i += 1

    os.system('mv *.csv %s' %cwd)
    os.chdir(cwd)

if __name__ == "__main__":
    main()

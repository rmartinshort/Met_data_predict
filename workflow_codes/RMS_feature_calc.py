#!/usr/bin/env python
#RMS 2018
#Alternative method for constucting features. In this case, we calculate one set of features per input DAS file, and
#generate dataframes for each of the features across all channels. These can then be directly linked to the soil
#moisture dataframe

import pandas as pd
import obspy as op
import numpy as np
from ts_tools import despike
import FeatureCalculate as FC
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

    FCCalc = FC.FeatureCalc()
    feature_names = FCCalc.return_feature_names()

    i = 0
    file_name_list = []
    for fname in fnames:

        print ('Calculating features for file %s' %fname)

        #Save the name of the file
        file_name_list.append(fname[2:])

        #Read in file
        f = op.read(fname,format='mseed')

        #LOOP OVER THE CHANNELS WE WANT TO EXTRACT, SELECT THE CHANNEL, CALCULATE FEATURES AND
        #APPEND TO FEATURE MATRIX

        j = 0
        for traceid in traces:

            trace = f[traceid]
            trace = despike(trace,scale_factor=100)
            inslice = trace.data
            #load the timeseries to the FCCalc object
            FCCalc.load_new_ts(inslice)

            if not np.isnan(inslice).any():

                fa = FCCalc.calculate_all_features()

                f1[i,j] = fa[0]
                f2[i,j] = fa[1]
                f3[i,j] = fa[2]
                f4[i,j] = fa[3]
                f5[i,j] = fa[4]
                f6[i,j] = fa[5]
                f7[i,j] = fa[6]
                f8[i,j] = fa[7]
                f9[i,j] = fa[8]
                f10[i,j] = fa[9]
                f11[i,j] = fa[10]
                f12[i,j] = fa[11]
                f13[i,j] = fa[12]
                f14[i,j] = fa[13]

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
    for matrix in [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14]:
        df = pd.DataFrame(matrix,columns=channel_names)
        df['file_name'] = file_name_list
        df.to_csv(feature_names[i]+'rms_fcalc.csv',index=False)
        i += 1

    os.system('mv *.csv %s' %cwd)
    os.chdir(cwd)

if __name__ == "__main__":
    main()

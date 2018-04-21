#!/usr/bin/env python
#RMS 2018

#Calulation of features using the FeatureCalculate class, with option for
#parallelization. This is currently set up to calculate one feature set per file
#Could be extended to determining features across sliding windows of files

import multiprocessing
import pandas as pd
import obspy as op
import numpy as np
from ts_tools import despike
import FeatureCalculate as FC
import os

def parallel_calc_features(queue,file_names_list,traces):

    '''Function to calculate features and fill a matrix. file_names_list is the result of splitting
    the full fnames list over nprocs
    '''

    FCCalc = FC.FeatureCalc()
    ltrace = len(file_names_list)
    nchannels = len(traces)
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
    fname_list = []

    i = 0
    for fname in file_names_list:
        print ('Calculating features for file %s' %fname)
        #Save the name of the file
        fname_list.append(fname[2:])
        #Read in file
        f = op.read(fname,format='mseed')

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

    #return the filled feature matrics to queue
    results = [fname_list,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14]
    queue.put(results)

def splitprocs(fnames,nprocs):

    '''Split the dataframe into chunks to be worked on by each process
    '''

    nfiles = len(fnames)
    files_per_proc = int(np.floor(nfiles/nprocs))
    segments = []
    lower = 0
    for i in range(0,nprocs):
        upper = lower + files_per_proc
        if upper >= nfiles:
            upper = nfiles

        if i == (nprocs-1):
            upper = nfiles

        #Append the upper and lower file number to the thread job count
        segments.append([lower,upper])
        lower = upper

    return segments


def main():

    #----------------------------------------------------------------------------------------------------
    #Prepare the dataset and extract the file names
    #---------------------------------------------------------------------------------------------------

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

    #---------------------------------------------------------------------------------------------------

    #numbered list of the trace indices to extract
    traces = range(75,95)

    #Number of procs
    nprocs = 2

    #Set up processing queue
    q = multiprocessing.Queue()
    nchannels = len(traces)

    #--------------------------------------------------------------------------------------------------
    #allocate full matrices to be write
    #Generate linking column
    lineCSN_M3 = lineCSN_M3.assign(file_name=lineCSN_M3['file_name'].apply(lambda x: x[2:]))
    ltrace = len(fnames)
    nchannels = len(traces)
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
    all_fnames = []
    mlist = [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14]
    #--------------------------------------------------------------------------------------------------

    #Get the names of the features we are calculating
    FCCalc = FC.FeatureCalc()
    feature_names = FCCalc.return_feature_names()

    #Location of the DAS data files
    datadir = "/media/rmartinshort/My Book/4Robert"
    cwd = os.getcwd()
    os.chdir(datadir)

    maxnprocs = multiprocessing.cpu_count()
    segments = splitprocs(fnames,nprocs)
    print segments
    processes = []

    for i in range(nprocs):
        fsegments = fnames[segments[i][0]:segments[i][1]]
        p = multiprocessing.Process(target=parallel_calc_features,args=(q,fsegments,traces))
        processes.append(p)
        p.start()

    #Get the results and fill the matrices we just made
    k = 0
    for p in processes:
        results = q.get()
        start_index = segments[k][0]
        end_index = segments[k][1]
        all_fnames = all_fnames + results[0]

        #fill the master matrix copies with data

        j = 1
        for matrix in mlist:
            matrix[start_index:end_index,:] = results[j]
            mlist[i] = matrix
            j += 1
        k += 1

    #Terminate the processes
    for p in processes:
        p.join()


    #--------------------------------------------------------------------------------------------------
    #Write to file (master process)
    #Note!! The file names might not get written in chronological order, but this shouldn't matter in the end
    #since we end up joining the dataframes together based on filename. We also attempt to fix this in the
    #next section
    #--------------------------------------------------------------------------------------------------

    #Create names vector for the columns
    channel_names = []
    for traceid in traces:
        name = 'channel_%i' %traceid
        channel_names.append(name)

    #Write each matrix to a pd dataframe and to outfile
    i = 0
    for matrix in mlist:
        df = pd.DataFrame(matrix,columns=channel_names)
        df['file_name'] = all_fnames

        #put the times in order
        ftstamp = df['file_name'].apply(lambda x: x.split('_')[0])
        df['ftstamp'] = ftstamp
        print(df.head())
        df.sort_values(by=['ftstamp'],inplace=True)
        df.drop('ftstamp',axis=1,inplace=True)

        df.to_csv(feature_names[i]+'_rms_fcalc_parallel.csv',index=False)
        i += 1

    os.system('mv *.csv %s' %cwd)
    os.chdir(cwd)


if __name__ == "__main__":

    main()

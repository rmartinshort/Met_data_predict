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

def parallel_calc_features(queue,file_names_list,traces,rolling_window_len,rolling_window_offset,len_trace_seconds):

    '''Function to calculate features and fill a matrix. file_names_list is the result of splitting
    the full fnames list over nprocs
    '''

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

        #Time marker corresponds the center of that trace. All traces in the merged stream
        #should be the same length, so we only need to do this calculation on one of them
        #This becomes the time stap at which we interpolate the soil moisture or temperature
        #timeseries
        tm = op.UTCDateTime(tr.stats.starttime + len_trace/2)
        time_markers.append(tm)

        if p > 0:
            print(time_markers[p] - time_markers[p-1])

        #return the filled feature matrics to queue
        results = [time_markers,f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14]
        queue.put(results)

def splitprocs(fnames,nprocs,rolling_window_offset):

    '''Split the dataframe into chunks to be worked on by each process
    '''

    #Total number of files that we can consider
    nfiles = lentrace = divmod(len(fnames),rolling_window_offset)[0]
    files_per_proc = int(np.floor(nfiles/nprocs))
    extra_files = divmod(files_per_proc,rolling_window_offset)[1]
    segments = []
    lower = 0
    for i in range(0,nprocs):

        #Each processor needs to know about some number of files 'beyond' its allocated list
        #because of the overlap
        upper = lower + files_per_proc + extra_files
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
    fnames = list(lineCSN['file_name'])[:100]

    #---------------------------------------------------------------------------------------------------

    #numbered list of the trace indices to extract
    traces = range(75,95)

    #rolling window charactersitics in terms of number of files
    rolling_window_len = 10
    rolling_window_offset = 5
    len_trace_seconds = rolling_window_len*60.0

    #Set up feature matrices
    ltrace = divmod(len(fnames),rolling_window_offset)[0] - 1
    nchannels = len(traces) #number of channels to extract

    #Number of procs
    nprocs = 2

    #Set up processing queue
    q = multiprocessing.Queue()
    nchannels = len(traces)

    #--------------------------------------------------------------------------------------------------
    #allocate full matrices to be write
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
    all_times = []
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
        p = multiprocessing.Process(target=parallel_calc_features,args=(q,fsegments,traces,rolling_window_len,
        rolling_window_offset,len_trace_seconds))
        processes.append(p)
        p.start()

    #Get the results and fill the matrices we just made
    k = 0
    for p in processes:
        results = q.get()
        start_index = segments[k][0]
        end_index = segments[k][1]
        all_times = all_times + results[0]

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
        df['time'] = all_times
        df.sort_values(by=['time'],inplace=True)
        df.to_csv(feature_names[i]+'_rms_fcalc_sliding_parallel.csv',index=False)
        i += 1

    os.system('mv *.csv %s' %cwd)
    os.chdir(cwd)


if __name__ == "__main__":

    main()

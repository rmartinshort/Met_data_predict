#!/usr/bin/env python
#RMS 2018
#Downsample the DAS data and rename the files. This could act as a platform to build further code upon

import multiprocessing
import obspy as op
import numpy as np
import glob
import os
import pandas as pd
import time

def main():

    nprocs = multiprocessing.cpu_count()
    #datadir = "/home/rmartinshort/Documents/Berkeley/Fiber/example_data/downsample_test_2"
    datadir = "/media/rmartinshort/My Book/4Robert"
    cwd = os.getcwd()
    os.chdir(datadir)

    #Need to do all processing in datadir to save space

    if not os.path.exists("allfilenames.dat"):
        # Generate a file that contains all the names of the .mseed files to be processed
        print("listing all file names")
        os.system("find -type f -name '*.mseed' > allfilenames.dat")

    else:
        print("allfilenames.dat found. Proceeding to load file and loop over data")


    fnames = pd.read_csv('allfilenames.dat',names=['fname'])
    segs = splitprocs(fnames,nprocs)

    jobs = []
    for i in range(nprocs):
       p = multiprocessing.Process(target=process,args=(i,segs[i],fnames))
       jobs.append(p)
       p.start()

def splitprocs(df,nprocs):

    "split the dataframe into chunks to be worked on by each thread"

    nfiles = len(df)
    files_per_proc = int(np.floor(nfiles/nprocs))
    print(files_per_proc)
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


def process(jobid,indices,fnames,resamp_val=40,rename=True):

    "MPI-style processing of the chunks"

    start = indices[0]
    stop = indices[1]

    workfiles = list(fnames['fname'])[start:stop]
    print(" I am process %i with %i files to process" %(jobid,len(workfiles)))

    for element in workfiles:
        time = '20'+str(element.split('_')[1])
        line = element.split('_')[2]
        nname = time+"_"+str(resamp_val)+"_Hz_"+line

        #Open the file and do processing
        st = op.read(element,format='mseed')
        #application of the low pass filter is important here to avoid 
        #interpolation between noise spikes
        st.filter('lowpass',freq=20.0,zerophase=True)
        try:
          st.interpolate(resamp_val)
        except:
          print("ERROR in interpolating %s" %nname)
          continue 
        #st.resample(resamp_val)
        st.write(nname,format='mseed')


if __name__ == "__main__":

    main()

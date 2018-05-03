#!/usr/bin/env python

#Tool to compute cross correlations along a line, determine some QC parameter that
#describes how good the cross correlation is at capturing an EGF, then write to a dataframe

import glob
import os
import numpy as np
import obspy as op
from obspy.signal.cross_correlation import correlate
import pandas as pd
from ts_tools import spectralWhitening, despike

def main():

    os.chdir("/home/rmartinshort/Documents/Berkeley/Fiber/data_test/Soil_moisture_predict")

    all_data = pd.read_csv("Fnames_soilM_temp.csv")
    lineEW = all_data[all_data['Line'] == "Line2EW"][['file_name','DateTime']]
    #In this experiment we're only interested in the file names and the times

    lineEW.reset_index(drop=True,inplace=True)
    lineEW.assign(DateTime = pd.to_datetime(lineEW['DateTime']).values)
    otime = op.UTCDateTime(lineEW['DateTime'][0])
    #lineEW.set_index('DateTime',inplace=True)

    #Generate linking column
    lineEW = lineEW.assign(file_name=lineEW['file_name'].apply(lambda x: x[2:]))
    fnames = lineEW['file_name']

    ###############################################################################
    # Enter data directory and do XC calculation

    datadir = "/media/rmartinshort/My Book/4Robert"
    cwd = os.getcwd()
    os.chdir(datadir)

    QC_xcorrs = np.full(len(fnames),np.nan)

    j = 0
    for infile in fnames.values:

        if j%1000 == 0:
            print(j)

        try:

            f = op.read(infile,format='mseed')
            f.detrend('demean')
            f.detrend('linear')

            shift = f[0].stats.npts
            nchan = len(f)
            f0 = whiten(f[nchan-1])
            fc = correlate(f0,f0,shift)
            X = np.linspace(-60,60,2*shift+1)

            #generate and fill cross correlation matrix
            Xcorr_mat = np.zeros([len(fc),nchan])
            Xcorr_mat[:,0] = fc

            for i in range(1,nchan):
                fc = correlate(f0,whiten(f[nchan-(i+1)]),shift)
                Xcorr_mat[:,i] = fc

            #QC parameter for determining quality of correlations
            QC_xcorrs[j] = sum(sum(abs(Xcorr_mat[shift-40:shift+40,:])))

        except:
            print("Error dealing with file %s" %infile)
            continue

        j += 1

    lineEW['XC_scores'] = QC_xcorrs
    os.chdir(cwd)
    lineEW.to_csv('Line_2EW_XC_scores.dat',index=False)


def whiten(trace):

    w_dat = spectralWhitening(trace.data)
    trace.data = w_dat

    return trace


if __name__ == "__main__":

    main()

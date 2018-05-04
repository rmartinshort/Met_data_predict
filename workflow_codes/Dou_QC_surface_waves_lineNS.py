#!/usr/bin/env python

#Data screening with an amplitude vs. offset metric, as described by Dou et al. 2017

import glob
import os
import numpy as np
import obspy as op
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

def func_powerlaw(x,m,c0,c1):
    '''
    We want to fit a power law of this type to the vectors of channel offset from west
    vs. rms noise value
    '''

    return c0 + c1*(x**m)

def main():

    os.chdir("/home/rmartinshort/Documents/Berkeley/Fiber/data_test/Soil_moisture_predict")

    all_data = pd.read_csv("Fnames_soilM_temp.csv")
    lineSN = all_data[all_data['Line'] == "LineCSN"][['file_name','DateTime']]
    #In this experiment we're only interested in the file names and the times

    lineSN.reset_index(drop=True,inplace=True)
    lineSN.assign(DateTime = pd.to_datetime(lineSN['DateTime']).values)
    otime = op.UTCDateTime(lineSN['DateTime'][0])
    #lineEW.set_index('DateTime',inplace=True)

    #Generate linking column
    lineSN = lineSN.assign(file_name=lineSN['file_name'].apply(lambda x: x[2:]))
    fnames = lineSN['file_name']

    ###############################################################################
    # Enter data directory and do XC calculation

    datadir = "/media/rmartinshort/My Book/4Robert"
    cwd = os.getcwd()
    os.chdir(datadir)

    QC_surf = np.full(len(fnames),np.nan)
    c_vals = np.full(len(fnames),np.nan)
    corr_coefs = np.full(len(fnames),np.nan)

    j = 0
    for infile in fnames.values:

        if j%1000 == 0:
            print(j)

        try:
            f = op.read(infile,format='mseed')

            #Calculate rms noise vector
            nchan = len(f)
            chan_n = np.zeros(nchan)
            rms_vals = np.zeros(nchan)

            for i in range(nchan):
                tr = f[i]
                tr.detrend('demean')
                tr.detrend('linear')
                x = f[nchan-(i+1)].data #we want to measure from W to E
                rms_noise = np.sqrt(np.mean(x*x))
                chan_n[i] = i
                rms_vals[i] = rms_noise

            popt, pcov = curve_fit(func_powerlaw,chan_n,rms_vals)
            corr_coef = pearsonr(rms_vals,func_powerlaw(chan_n,*popt))[0]
            optval = abs(popt[0]*corr_coef + 2)

            QC_surf[j] = optval
            c_vals[j] = popt[0]
            corr_coefs[j] = corr_coef

        except:
            print("Error dealing with file %s" %infile)
            continue

        j += 1

    lineSN['QC_scores'] = QC_surf
    lineSN['c_vals'] = c_vals
    lineSN['corr_coefs'] = corr_coefs
    os.chdir(cwd)
    lineSN.to_csv('Line_CSN_QC_scores.dat',index=False)

if __name__ == "__main__":

    main()

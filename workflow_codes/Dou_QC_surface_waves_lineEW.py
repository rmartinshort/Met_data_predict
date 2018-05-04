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

    QC_surf = np.full(len(fnames),np.nan)
    c_vals = np.full(len(fnames),np.nan)
    corr_coefs = np.full(len(fnames),np.nan)

    j = 0
    for infile in fnames.values:

        if j%1000 == 0:
            print(j)

        try:
            f = op.read(infile,format='mseed')
            rev_f = list(reversed(f))

            #Calculate rms noise vector
            nchan = len(f)
            chan_n = np.zeros(nchan)
            rms_vals = np.zeros(nchan)

            for i in range(nchan):
                tr = rev_f[i]
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

    lineEW['QC_scores'] = QC_surf
    lineEW['c_vals'] = c_vals
    lineEW['corr_coefs'] = corr_coefs
    os.chdir(cwd)
    lineEW.to_csv('Line_2EW_QC_scores.dat',index=False)

if __name__ == "__main__":

    main()

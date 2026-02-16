import os
import numpy as np
import json
import pickle as pkl
import matplotlib.pyplot as plt
import gc as gc
from astropy.table import Table
from s3fit import FitFrame
from tqdm import tqdm
from .s3fit_fit import s3fit_analysis



class s3fit_batch_analysis:
    def __init__(self, input_flist, outsuff=None, nmock=1, 
                 data_resolution=3500, 
                 use_phot=False, phot_filterdir=None, 
                 nfiles=None, runfit=False):
        
        # initiate attributes
        self.input_flist=input_flist

        if outsuff is None:
            self.outsuff=''
        else:
            self.outsuff=f'_{outsuff}'

        self.nmock=nmock

        # spectral status
        self.data_resolution=data_resolution
        
        # photometry filter directory
        self.use_phot=use_phot
        self.phot_filterdir=phot_filterdir

        # I/O stuff
        self.nfiles=nfiles
        self.runfit=runfit

        
        # initialization checks
        if self.use_phot is True:
            assert os.path.isdir(self.phot_filterdir), f'{self.phot_filterdir} is not a directory'

        # unpack file list
        self.package=self._read_flist()

    def _read_flist(self):
        with open(self.input_flist, 'r') as f:
            filelist=[i.strip('\n') for i in f.readlines() if i[0]!='#']

        filelist=[i.split(' ') for i in filelist]
        return filelist


    def _run_fit_single(self, idata, icfg, ofile):
        
        s3fit_obj=s3fit_analysis(idata, icfg, 
                                 self.data_resolution, 
                                 outputfile=ofile,
                                 use_phot=self.use_phot, phot_filtersdir=self.phot_filterdir, 
                                 nmocks=self.nmock,)
        if self.runfit:
            s3fit_obj.run_fitting()
        else:
            print('Skip fitting steps')
        

    def run(self):
        if self.nfiles is None:
            nfiles=len(self.package)
        else:
            nfiles=self.nfiles
            
        # nfiles it run
        for i in tqdm(range(0,nfiles)):
            
            idat, icfg = self.package[i]
            outdir=os.path.dirname(icfg)
            resfile=os.path.basename(icfg).replace('config.cfg', 'res.pkl')
            errfile=os.path.basename(icfg).replace('config.cfg', 'res.err')
            outpath=os.path.join(outdir, resfile)
            outerror=os.path.join(outdir, errfile)

            # set up s3fit object
            print(f'Fitting {icfg}')

            try:
                self._run_fit_single(idat, icfg, outpath)
            except Exception as e: 
                err_txt=e
                with open(outerror, 'w') as f:
                    f.write(e)
                
                
            break
            
            
            
            

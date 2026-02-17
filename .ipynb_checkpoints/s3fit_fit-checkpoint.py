
import os
import numpy as np
import json
import pickle as pkl
import matplotlib.pyplot as plt
import gc as gc
from astropy.table import Table
from s3fit import FitFrame
from .s3fit_plots import plot_spectral
#import s3fit as s3fit

class s3fit_analysis:
    def __init__(self, input_dataset, input_config, resolution, outputfile='./s3fit_res.pkl',
                 use_phot=True, phot_filtersdir=None,
                 redshift=None, spec_mask=None, nmocks=1, usemulticore=True,
                 verbose=False):

        # settings initiation
        self.input_dataset=input_dataset
        self.input_config=input_config
        self.outputfile=outputfile

        # I/O settings 
        self.use_phot=use_phot
        self.phot_filtersdir=phot_filtersdir

        # real initiations
        
        # load datasets & config files
        self.dataset = self._load_datasets()
        self.cfgdict = self._load_config()
    
        # set object names 
        self.object_name=self.dataset['name']
        # set the redshift vales 
        if redshift is None:
            self.redshift=self.dataset['redshift']
        else:
            assert type(redshift) is float, "Redshift must be a float"
            self.redshift=redshift

        # set the input datasets
        self.specdata = self.dataset['specdata']
        if use_phot:
            self.photdata = self.dataset['photdata']
        else:
            self.photdata ={'band': None, 'fval': None, 'func': None}
        
        # set spectrograph resolution
        self.resolution=[resolution for i in self.specdata['wave']]
        self.nmock=nmocks
        

        # initiate s3fit boject
        self.FF = FitFrame(spec_wave_w=self.specdata['wave'], 
                           spec_flux_w=self.specdata['fval'], spec_ferr_w=self.specdata['unc'],
                           spec_R_inst_w=self.resolution, spec_valid_range=None, 
                           phot_name_b=self.photdata['band'], 
                           phot_flux_b=self.photdata['fval'], phot_ferr_b=self.photdata['func'], 
                           phot_flux_unit='mJy', phot_trans_dir=self.phot_filtersdir, 
                           v0_redshift=self.redshift, model_config=self.cfgdict, 
                           num_mocks=self.nmock, fit_grid='log',
                           print_step=verbose, plot_step=False, canvas=None, 
                           use_multi_thread=usemulticore)
            

    def run_fitting(self):
        print('S3Fit Go brrr....')
        fig, axs=plt.subplots(2,1, figsize=(18,8), dpi=75)
        
        self.FF.canvas = (fig,axs)
        self.FF.main_fit()

        self._dump_results()
        
    def _load_datasets(self):
        with open(self.input_dataset, 'rb') as file:
            dataset = pkl.load(file)
        return dataset

    def _load_config(self):
        with open(self.input_config) as json_data:
            cfgdict= json.load(json_data)
        return cfgdict

    def _dump_results(self):
        output_package = {'name': self.object_name, 'redshift': self.redshift,
                          'specdata':self.specdata, 'photdata':self.photdata,
                          'bestfit_prop':None, 's3fit_output': self.FF.output_mc}
        
        with open(self.outputfile, 'wb') as handle:
            pkl.dump(output_package, handle, protocol=pkl.HIGHEST_PROTOCOL)

    def plot_dataset(self):
        # plot data first before running
        out_data_fig=self.outputfile.replace('.pkl','.pdf')
        specfig=plot_spectral(self.specdata, self.photdata,)
        specfig.savefig(out_data_fig)
        
        

        

        
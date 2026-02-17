import os
import glob
import sys
import json
import shutil
import numpy as np
import pickle as pkl
from astropy.table import Table
from tqdm import  tqdm

class s3fit_genconfig:
    def __init__(self, redshift, ssp_mod_file, add_eml=True, eml_line_dict={},
                 use_pyneb=False, **kwargs):

        # define the source redshift
        self.redshift=redshift
        
         # ssp model library
        self.ssp_model_file=ssp_mod_file
        
        # eml Model parameters 
        self.add_eml = add_eml
        self.eml_line_dict = eml_line_dict
        self.use_pyneb = use_pyneb

        # create SSP config
        self.ssp_config=self.init_ssp_config(**kwargs)
        self.eml_config=self.init_eml_config(eml_dict=self.eml_line_dict,)
        
        model_config = {'stellar': {'enable': True, 'config': self.ssp_config,   'file': self.ssp_model_file},
                        'line'   : {'enable': self.add_eml, 'config': self.eml_config, 'use_pyneb': self.use_pyneb}}
                        #'agn'  : {'enable': True, 'config': agn_config}, 
                        #'torus': {'enable': True, 'config': torus_config, 'file': torus_file}}
                
        self.s3f_config=model_config
    
    def init_eml_config(self, eml_dict, 
                        eml_nrl_vshift_par=[-100,100, 'free'], 
                        eml_nrl_fwhm_par=[10, 1000, 'free'], 
                        eml_nrl_av_par=[0, 5, 'free'], 
                        eml_nrl_edensity_par=[1.3, 4.3, 'free'], 
                        eml_nrletemp_par=[4, None, 'fix'], 
                        add_blr_line=False, blr_names=['Ha'],
                        eml_brl_vshift_par=[-100,100, 'free'], 
                        eml_blr_fwhm_par=[1000,8000, 'free'],
                        eml_blr_av_par=[0, 1, 'free'],
                        eml_blr_edensity_par=[1.3, 4.3, 'free'], 
                        eml_blr_etemp_par=[4, None, 'fix'], ):

        nlr_config = {'pars': [eml_nrl_vshift_par, # velocity shift (km/s)
                               eml_nrl_fwhm_par, # velocity FWHM (km/s)
                               eml_nrl_av_par, # extinction (AV)
                               eml_nrl_edensity_par, # electron density (log cm-3)
                               eml_nrletemp_par, # electron temperature (log K)
                              ],
                              'info': {}} # use all default setups 
 
        
        #nlr_config = {'NLR': {'pars': {'par1': self._map_pars(eml_vshift_par), 
        #                               'par2': self._map_pars(eml_narrowfwhm_par),
        #                               'par3': self._map_pars(eml_av_par),
        #                               'par4': self._map_pars(eml_edensity_par),
        #                               'par5': self._map_pars(eml_etemp_par)},
        #                        'info': {'line_used': ['all']}}}

        eml_config={'NLR': nlr_config}
        
        if add_blr_line:
            blr_config = {'pars': [eml_brl_vshift_par, # velocity shift (km/s)
                                   eml_blr_fwhm_par, # velocity FWHM (km/s)
                                   eml_blr_av_par, # extinction (AV)
                                   eml_blr_edensity_par, # electron density (log cm-3)
                                   eml_blr_etemp_par, # electron temperature (log K)
                                  ],
                                  'info': {'line_used' : 'BLR'}} # use all default setups 

            eml_config['BLR']=blr_config

        
        return eml_config


    def init_ssp_config(self, ssp_sfh_model='delayed', ssp_age_min=-3, ssp_age_max='universe', ssp_metal_val=0.02,
                        old_vshift_par=[-800,800,'free'], old_fwhm_par=[100,1200,'free'], old_av_par=[0.,2.5,'free'], 
                        old_csp_par=[-6,0.7,'free'], old_tau_par=[-1, 0,'free'], 
                        add_young=False, 
                        young_vshift_par=[-800,800,'free'], young_fwhm_par=[100,1200,'free'], young_av_par=[0.,2.5,'free'], 
                        young_csp_par=[-3,-0.7,'free'], young_tau_par=[-1.3, -0.3,'free'], ):

        """
        Single spectral model settings 
        Global Parameters :
            - ssp_age_min :min SSP age (log Gyr)
            - ssp_age_max :max SSP age, can be either given in log Gyr, or in the universe age at the given v0_redshift
            - ssp_metal_val :metallicity, can be 'all', 'solar', or any combination of [0.004,0.008,0.02,0.05]
            - ssp_sfh_model :name of SFH function
        SSP Model parameters
            - *_vshift_par : velocity shift (km/s)
            - *_fwhm_par : velocity FWHM (km/s)
            - *_av_par : extinction (AV)
            - *_csp_par : CSP age of old population (or galaxy age) (log Gyr)
            - *_tau_par : declining timescale of exponential or delayed SFH (log Gyr)

        """
        # run input checks
        if isinstance(ssp_sfh_model,str):
            assert ssp_sfh_model in ['nonparametric', 'exponential', 'delayed', 'constant'], 'SFH model not in list'
            if ssp_sfh_model=='nonparametric':
                assert add_young is False, 'Non-parameteric SFH can only be used with a single SFH'
            
        else:
            assert callable(ssp_sfh_model), 'Input SFH model is not a valid function'
        

        
        # init SSP config    
        ssp_config={'main': {'pars': {'voff': self._map_pars(old_vshift_par), 
                                     'fwhm': self._map_pars(old_fwhm_par), 
                                     'Av': self._map_pars(old_av_par), 
                                     'log_csp_age' : self._map_pars(old_csp_par), 
                                     'log_csp_tau' : self._map_pars(old_tau_par), 
                                    }, 
                            'info': {'log_ssp_age_min' : ssp_age_min,   
                                     'log_ssp_age_max' : 'universe',    
                                     'ssp_metallicity' : ssp_metal_val, 
                                     'sfh_name'        : ssp_sfh_model, 
                                    },}}
        if add_young:
            ssp_config['young']={'pars': {'voff': self._map_pars(young_vshift_par), 
                                     'fwhm': self._map_pars(young_fwhm_par), 
                                     'Av': self._map_pars(young_av_par), 
                                     'log_csp_age' : self._map_pars(young_csp_par), 
                                     'log_csp_tau' : self._map_pars(young_tau_par), 
                                    }, 
                                 'info': {'log_ssp_age_min' : ssp_age_min,   
                                          'log_ssp_age_max' : 'universe',    
                                          'ssp_metallicity' : ssp_metal_val, 
                                          'sfh_name'        : ssp_sfh_model, 
                                          },}

        return ssp_config
            

    def _map_pars(self, ipar):
        # test data type 
        llim,ulim,fix=ipar
        return {'min': llim, 'max': ulim, 'tie': fix}

    def dump_config(self, outconfig='./test_config.json'):
        with open(outconfig, 'w') as f:
            json.dump(self.s3f_config, f)
    


class s3fit_genconfig_batch:
    def __init__(self, input_dir, output_dir, sspmod, outflist='./filelist.list',
                 add_eml=True, eml_lib=None, 
                 ssp_sfh='nonparametric', 
                 add_young=False,
                 suff=None, ):
        
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.sspmod = os.path.abspath(sspmod)  
        self.outflist=outflist
        # emission line parameter
        self.add_eml=add_eml
        self.eml_lib=eml_lib
        self.sfh=ssp_sfh
        self.add_young=add_young

        # output suffix
        if suff is None:
            self.suff=''
        else:
            self.suff=f'_{suff}'

        # start proc
        self.flist=glob.glob(os.path.join(self.input_dir,'*.pkl'))
        self.flist.sort(reverse=True)


    def generate_configs(self):
        # loop through pkl and generate the output
        os.makedirs(self.output_dir,exist_ok=True)
        flist_list=[]
        for i in tqdm(range(0, len(self.flist))):
            input_pkl=self.flist[i]

            with open(input_pkl, 'rb') as file:
                dataset = pkl.load(file)

            name, zspec = dataset['name'], dataset['redshift']
            #print(dataset['name'])
            s3f_cfg_obj = s3fit_genconfig(zspec, 
                                          self.sspmod, 
                                          ssp_sfh_model=self.sfh, 
                                          eml_line_dict=self.eml_lib, 
                                          add_eml=self.add_eml,  
                                          add_young=self.add_young)

            s3f_config = s3f_cfg_obj.s3f_config

            # define output files
            des_path=os.path.join(self.output_dir, name)
            dat_file=os.path.basename(input_pkl)
            cfg_file=dat_file.replace('_dataset.pkl',f'{self.suff}_config.cfg')
            obj_dat_path=os.path.join(des_path, dat_file)
            obj_cfg_path=os.path.join(des_path, cfg_file)

            # copy files
            os.makedirs(des_path, exist_ok=True)
            self._copy_files(input_pkl, obj_dat_path) 
            s3f_cfg_obj.dump_config(outconfig=obj_cfg_path)
            

            package=(os.path.abspath(obj_dat_path),os.path.abspath(obj_cfg_path))
            flist_list.append((package))

        flist_dir=os.path.dirname(self.outflist)
        os.makedirs(flist_dir, exist_ok=True)
        flist_table=Table(rows=flist_list, names=['input_dataset','config_file'])
        flist_table.write(self.outflist, format='ascii.commented_header', overwrite=True)
        return flist_table


    def _copy_files(self,src,des):
        try:
            shutil.copy2(src, des)
        except FileNotFoundError:
            print(f"Error: Source file '{src}' not found.")
        except PermissionError:
            print(f"Error: Permission denied when accessing files.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")


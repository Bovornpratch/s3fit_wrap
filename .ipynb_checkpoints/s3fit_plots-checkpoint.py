import matplotlib.pyplot as plt



def plot_spectral(specdata=None, photdata=None, figsize=(18,6), dpi=75):

    
    fig=plt.figure(figsize=figsize, dpi=dpi)
    ax1=fig.add_subplot(111)
    ax1.plot(specdata['wave'], specdata['fval']*1e19)
        
    ax1.minorticks_on()
    ax1.tick_params(labelsize=18, direction='in', length=10, right=True, top=True)
    ax1.tick_params(which='minor', length=5, direction='in', right=True, top=True)
    ax1.set_xlabel('Wavelength ($\AA$)', fontsize=20)
    ax1.set_ylabel('$f_{\lambda}\ \mathrm{(10^{-19}\ erg s^{-1} cm^{-2} \AA^{-1})}$', fontsize=20)
        
    fig.tight_layout()   

    return fig
        
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import wofz #Faddeeva function 

def Gaussian(E, E0, FWHM):
    x = (E-E0)*2/FWHM
    return np.exp(-np.log(2)*x**2)

def Lorentzian(E, E0, FWHM):
    x = (E-E0)*2/FWHM
    return 1 / (1 + x**2)

def Voigt(E, E0, FWHM):
    """
       FWHM as [FWHM_G, FWHM_L]
    """
    gamma, sigma = [FWHM[0],FWHM[1]/np.sqrt(2*np.log(2))]
    z = (E - E0 + 1j*gamma)/(sigma*np.sqrt(2))
    return wofz(z).real/(sigma*np.sqrt(2*np.pi))

def Voigt1(E, E0, FWHM):
    """
       FWHM as [FWHM_G, FWHM_L]
       Citation: He, Jian, and Qingguo Zhang, Journal of Optics A: 
                 Pure and Applied Optics 9.7 (2007): 565
    """
    fp2 = np.log(2)/FWHM[0]**2
    fp = np.sqrt(fp2)
    x = (E-E0)*fp
    y = FWHM[1]*fp
    z = (E+E0)*fp
    fourPi = 4*np.pi
    return (fp/(2*np.sqrt(np.pi)))* \
                (np.exp(fourPi*y**2-z**2)*np.cos(fourPi*y*z)+
                 np.exp(fourPi*y**2-x**2)*np.cos(fourPi*y*x)) 

def spectrum_boardening(E, I, E_range=None, resol=None, lineshape='G',
                        FWHM=0.1, shift=0.0, normalize=True):
    """
      Function to do spectrum_boarding specified as lineshape.
    """
    assert callable(lineshape) or isinstance(lineshape, str)
    if callable(lineshape):
        lineshape_func = lineshape
    else:
        if lineshape in ['G', 'g', 'Gaussian', 'Gaussian']:
            lineshape_func = Gaussian
        elif lineshape in ['L', 'l', 'Lorentzian', 'lorentzian']:
            lineshape_func = Lorentzian
        elif lineshape in ['V','v','Voigt','voigt']:
            lineshape_func = Voigt
            assert len(FWHM) == 2
        else:
            raise ValueError('no lineshape function are found')
    
    assert len(E) == len(I)
    
    FWHM_min = min(FWHM) if isinstance(FWHM,list) else FWHM
    if(resol is None): resol = FWHM_min / 10
    if(E_range is not None):
        assert isinstance(E_range, tuple) or isinstance(E_range, list)
        assert len(E_range) ==2
    else:
        E_range = (np.min(E)-FWHM_min, np.max(E)+FWHM_min+resol)

    Ex = np.arange(E_range[0], E_range[1], resol)
    Iy = np.zeros(len(Ex))

    for i in range(len(E)):
        Iy += I[i]*lineshape_func(Ex, E[i]-shift, FWHM)

    if(normalize):
        Iy = Iy / max(Iy)

    return Ex, Iy

def plot_spectrum(E,I,data=None,E_range=None,color='C7',linewidth=3, 
                  label=None,ax=None,
                  broadening=True,shift=0.,lineshape='G',output_resol=None,
                  FWHM=1.0,normalize=False,
                  fill_valence=False,E_HOMO=None,fill_color=None,fill_alpha=0.5):

    """
      plot spectrum with input  
    """
    if data is None:
        plt_E = E
        plt_I = I
    else:
        assert isinstance(data, pd.DataFrame)
        assert isinstance(E,str)
        assert isinstance(I,str)
        plt_E = data[E]
        plt_I = data[I]
    
    # get data range
    E_interval = plt_E[1] - plt_E[0] if output_resol is None else output_resol
    if E_range is None: E_range = (min(plt_E), max(plt_E)+E_interval)

    # broadening 
    if broadening:
        plt_E, plt_I = spectrum_boardening(plt_E, plt_I, E_range,resol=output_resol,
                                           lineshape=lineshape,FWHM=FWHM,shift=shift,
                                           normalize=normalize)

    if ax is None:
        fig,ax =plt.subplots(figsize=(8,6))
    
    # plot curve
    ax.plot(plt_E, plt_I, label=label, color=color, linewidth=linewidth)
    
    # fill valence area for DOS
    if fill_valence:
        assert E_HOMO is not None
        E_HOMO_idx = np.argmax(plt_E > E_HOMO) if plt_E[-1] >= E_HOMO else -1 
        if(fill_color is None): fill_color = color
        ax.fill_between(plt_E[:E_HOMO_idx], plt_I[:E_HOMO_idx], 
                        color=fill_color, alpha=fill_alpha)
        
    return ax

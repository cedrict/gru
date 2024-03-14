import numpy as np
from creep_parameters import *
from common import *
import scipy.optimize as optimize   # FG for Newton Raphson

###################################################################################################
# definition of newton raphson function
# f(x)=sr - (sr_dis+sr_diff+sr_gbs+sr_lowT) with x= shear stress
###################################################################################################

def f(x,sr,gs,T):
    sr_dis=Adis*np.exp(-Edis/(Rgas*T))*x**ndis
    sr_diff=Adiff*np.exp(-Ediff/(Rgas*T))* x**ndiff * gs**(-mdiff)
    sr_gbs=Agbs*np.exp(-Egbs/(Rgas*T))* x**ngbs * gs**(-mgbs)
    sr_lowT=0
    if T< TlowT: sr_lowT=AlowT*np.exp(-ElowT/(Rgas*T) *(1-(x/taulowT)**plowT)**qlowT)
    val=sr-sr_dis-sr_diff-sr_gbs-sr_lowT
    return val

###################################################################################################
# same function but with strain rate for each mechanism as output
###################################################################################################

def compute_sr(x,sr,gs,T):
    sr_dis=Adis*np.exp(-Edis/(Rgas*T))*x**ndis
    sr_diff=Adiff*np.exp(-Ediff/(Rgas*T))* x**ndiff * gs**(-mdiff)
    sr_gbs=Agbs*np.exp(-Egbs/(Rgas*T))* x**ngbs * gs**(-mgbs)
    sr_lowT=0.
    if T< TlowT:sr_lowT=AlowT*np.exp(-ElowT/(Rgas*T) *(1-(x/taulowT)**plowT)**qlowT)
    return sr_dis,sr_diff,sr_gbs,sr_lowT

###################################################################################################
# this function returns the effective viscosity.
# note that strain rates are limited > 1e-19, and grain size > 20 microns
# and returned viscosity is such that it remains between 1e18 and 1e26
###################################################################################################

def viscosity(ee,T,imat,grainsize):  

        sr=max(1e-19,ee) # minimum strain rate (s^-1)
        gs=max(grainsize,20) # minimum grain size (microns)

        sigdis=(sr/Adis)**(1/ndis) * np.exp(Edis/ (ndis*Rgas*T))

        sigdiff=(sr/Adiff)**(1/ndiff) * gs**(mdiff/ndiff) * np.exp(Ediff/(ndiff*Rgas*T))

        siggbs=(sr/Agbs)**(1/ngbs) * gs**(mgbs/ngbs) * np.exp(Egbs / (ngbs*Rgas*T))

        if T<TlowT:
           siglowT=taulowT*(1-(-Rgas*T/ElowT * np.log(sr/AlowT))**(1/qlowT))**(1/plowT)
           sig=min(sigdis,sigdiff,siggbs,siglowT)
        else:
           sig=min(sigdis,sigdiff,siggbs)

        # NewtonRaphson Loop
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.newton.html
        taunr = optimize.newton(f,sig,args=(sr,gs,T),tol=1e-3, maxiter=100,disp=True)

        etaeff=taunr*1e6/2/sr #stress is in MPa
        
        computed_sr = compute_sr(taunr,sr,gs,T) # returns: sr_dis,sr_diff,sr_gbs,sr_lowT

        #viscosity cutoffs
        etaeff=min(etaeff,1e26)
        etaeff=max(etaeff,1e18)

        etaeff=1e21

        return etaeff,computed_sr[0],computed_sr[1],computed_sr[2],computed_sr[3] 

###################################################################################################

def gs_evolution(gs,sr,dinf,egs):

   

    return gs



###################################################################################################

def compute_dinf_gbs_diff(sr,T):
    eeq=sr/2
    expo=mdiff/ndiff-mgbs/ngbs
    db12=( (eeq/Agbs)**(1/ngbs)*(eeq/Adiff)**(-1/ndiff)*np.exp((Egbs/ngbs-Ediff/ndiff)/Rgas/T))**(1/expo)
    dinf=max(9,db12)
    return dinf

def compute_dinf_dis_diff(sr,T):
    eeq=sr/2
    expo=ndiff/mdiff
    db12=( (eeq/Adis)**(1/ndis)*(eeq/Adiff)**(-1/ndiff)*np.exp((Edis/ndis-Ediff/ndiff)/Rgas/T))**expo
    dinf=max(9,db12)
    return dinf

def compute_dinf(sr,T):
    eeq=sr/2
    dinf1=compute_dinf_gbs_diff(sr,T)
    tau_gbs=(eeq/Agbs *np.exp(Egbs/Rgas/T) *dinf1**mgbs) **(1/ngbs)
    dinf2=compute_dinf_dis_diff(sr,T)
    tau_dis=(eeq/Adis*np.exp(Edis/Rgas/T))**(1/ndis)
    if tau_gbs<tau_dis:
       dinf=dinf1
    else:
       dinf=dinf2
    return dinf 

###################################################################################################


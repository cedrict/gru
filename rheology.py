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

def viscosity(x,y,ee,T,imat,grainsize,egs):  

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

        # Find position of largest strain rate, this is the dominant mechanism
        # Translation key: dis = 0, diff = 1, gbs = 2, lowT = 3
        #max_mech = np.argmax(computed_sr)

        ########################################
        # define the final grain size
        ########################################
        A1=Agbs; n1=ngbs; E1=Egbs; m1=mgbs
        A2=Adiff; n2=ndiff; E2=Ediff; m2=mdiff
        eeq=sr/2
        db12=(eeq/A2*(A1/eeq)**(n2/n1) * np.exp(E2/Rgas/(T)-n2*E1/n1/Rgas/(T)))**(1/(n2/n1*m1-m2))
        if computed_sr[1]>computed_sr[2]:  # diffusion creep dominant over gbs
            A1=Adis; n1=ndis; E1=Edis; m1=0
            A2=Adiff; n2=ndiff; E2=Ediff; m2=mdiff
            eeq=sr/2
            db12=(eeq/A2*(A1/eeq)**(n2/n1) * np.exp(E2/Rgas/(T)-n2*E1/n1/Rgas/(T)))**(1/(n2/n1*m1-m2))
        dinf=max(10,db12)
        #dinf=20
        kin=4e13*egs
        if imat==3: kin=0
        grainsize=grainsize-sr*kin*(grainsize-dinf)
        ########################################


        #viscosity cutoffs
        etaeff=min(etaeff,1e26)
        etaeff=max(etaeff,1e18)

        return etaeff,grainsize,computed_sr[0],computed_sr[1],computed_sr[2],computed_sr[3] 

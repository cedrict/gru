import numpy as np
from creep_parameters import *
from common import *
import scipy.optimize as optimize   # FG for Newton Raphson

##################################################################
# definition of newton raphson function
# f(x)=sr - (sr_dis+sr_diff+sr_gbs+sr_lowT) with x= shear stress
##################################################################

def f(x,sr,gs,T):
    sr_dis=Adis*np.exp(-Edis/(Rgas*(T+273)))*x**ndis
    sr_diff=Adiff*np.exp(-Ediff/(Rgas*(T+273)))* x**ndiff * gs**(-mdiff)
    sr_gbs=Agbs*np.exp(-Egbs/(Rgas*(T+273)))* x**ngbs * gs**(-mgbs)
    sr_lowT=0
    if T<= TlowT: sr_lowT=AlowT*np.exp(-ElowT/(Rgas*(T+273)) * (1-(x/taulowT)**plowT)**qlowT)
    val=sr-sr_dis-sr_diff-sr_gbs-sr_lowT
    return val


def f(x,sr,gs,T):
    srdis=Adis*np.exp(-Edis/(Rgas*T))*x**ndis
    srdiff=Adiff*np.exp(-Ediff/(Rgas*T))* x**ndiff * gs**(-mdiff)
    srgbs=Agbs*np.exp(-Egbs/(Rgas*T))* x**ngbs * gs**(-mgbs)
    srlowT=0
    if T< TlowT: srlowT=AlowT*np.exp(-ElowT/(Rgas*T) * (1-(x/taulowT)**plowT)**qlowT)
    val=sr-srdis-srdiff-srgbs-srlowT
    return val

##################################################################
# same function but with strain rate for each mechanism as output
##################################################################

def compute_sr(x,sr,gs,T):
    sr_dis=Adis*np.exp(-Edis/(Rgas*(T+273)))*x**ndis
    sr_diff=Adiff*np.exp(-Ediff/(Rgas*(T+273)))* x**ndiff * gs**(-mdiff)
    sr_gbs=Agbs*np.exp(-Egbs/(Rgas*(T+273)))* x**ngbs * gs**(-mgbs)
    sr_lowT=0.
    if T<= TlowT:sr_lowT=AlowT*np.exp(-ElowT/(Rgas*(T+273)) * (1-(x/taulowT)**plowT)**qlowT)
    return sr_dis,sr_diff,sr_gbs,sr_lowT


def compute_sr(x,sr,gs,T):
    sr_dis=Adis*np.exp(-Edis/(Rgas*(T+273)))*x**ndis
    sr_diff=Adiff*np.exp(-Ediff/(Rgas*(T+273)))* x**ndiff * gs**(-mdiff)
    sr_gbs=Agbs*np.exp(-Egbs/(Rgas*(T+273)))* x**ngbs * gs**(-mgbs)
    sr_lowT=0.
    if T< TlowT:sr_lowT=AlowT*np.exp(-ElowT/(Rgas*(T+273)) * (1-(x/taulowT)**plowT)**qlowT)
    return sr_dis,sr_diff,sr_gbs,sr_lowT

#------------------------------------------------------------------------------
def viscosity(x,y,ee,T,imat,grainsize,egs):  # add grainsize as argument in viscosity FG
        # FG use experiment 1 for grain size rheology
        sr=max(1e-19,ee) # minimum strain rate
        gs=max(grainsize,20) # minimum grain size
        sigdis=(sr/Adis)**(1/ndis) * np.exp(Edis/ (ndis*Rgas*T))
        sigdiff=(sr/Adiff)**(1/ndiff) * gs**(mdiff/ndiff) * np.exp(Ediff/(ndiff*Rgas*T))
        siggbs=(sr/Agbs)**(1/ngbs) * gs**(mgbs/ngbs) * np.exp(Egbs / (ngbs*Rgas*T))
        siglowT=taulowT*(1-(-Rgas*T/ElowT * np.log(sr/AlowT))**(1/qlowT))**(1/plowT)
        sig=min(sigdis,sigdiff,siggbs)
        #print("gs, sigdis/gbs "+str(gs)+" "+str(sigdis)+" "+str(siggbs))
        if T<TlowT: sig=min(sigdis,sigdiff,siggbs,siglowT)
        # NewtonRaphson Loop
        taunr = optimize.newton(f,sig,args=(sr,gs,T),tol=1e-3, maxiter=20,fprime=None,fprime2=None)
        val=taunr*1e6/2/sr
        ##
        computed_sr = compute_sr(taunr,sr,gs,T) 
        # Find position of largest strain rate, this is the dominant mechanism
        # Translation key: dis = 0, diff = 1, gbs = 2, lowT = 3
        #max_mech = np.argmax(computed_sr)
        #####
        # define the final grain size
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
        #grainsize=(1-0.9*sr/1e-15)*grainsize
        # need to define them because output...
        #viscosity cutoffs
        val=min(val,1e26)
        val=max(val,1e18)
        return val,grainsize 

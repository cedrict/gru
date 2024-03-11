import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize
#import re
#import seaborn as sns
#import scipy.interpolate
from os import listdir
from os.path import isfile, join
from creep_parameters import * 

##################################################################################################
#####
defmap=True
rootdir = '/Users/fred/Library/CloudStorage/OneDrive-umontpellier.fr/0-Modelisation/STONE/StrainLoc/TEST/Histfile/'
# for histories as a function of temperature
nhist=10 # number of hist variable to store (temp and % fracturation)
n_dir=len(os.listdir(rootdir))  # nu
hist=np.zeros((nhist,n_dir-1))   # to be modified with nhist
ihist=0
files=glob.glob(rootdir+'Hist*')   # all the histfile
# quantities dependant on temperature
temperature=np.zeros(len(files))
weakening=np.zeros(len(files))
strainloc=np.zeros(len(files))
eps1=np.zeros(len(files))
eps2=np.zeros(len(files))
eps=0



if defmap==True:

##################################################################

    sr = 1e-15 # strain rate (s^-1)

    Tmin=380   # temperature (C) 
    Tmax=1300  # temperature (C)

    dmin=10    # grain size (microns)
    dmax=1e4   # grain size (microns)
    
    ##################################################################
    # build temperature array(s) 
    ##################################################################
    dT=2.5
    temp= np.arange(Tmin,Tmax,dT,dtype=np.float64) 
    ntemp=len(temp)
    
    ipllabel=[400,500,600,700,800,900,1000,1100,1200,1300]
    
    ##################################################################
    # grain size values array
    ##################################################################
    
    nd=500
    d= np.linspace(np.log10(dmin),np.log10(dmax),nd,dtype=np.float64) 
    d=10**d
    
    ###################################################################
    # allocate arrays for the boundaries between deformation mechanisms
    
    taubdis=np.zeros(ntemp,dtype=np.float64)
    dbdis=np.zeros(ntemp,dtype=np.float64)
    taubdiff=np.zeros(ntemp,dtype=np.float64)
    dbdiff=np.zeros(ntemp,dtype=np.float64)
    taubgbs=np.zeros(ntemp,dtype=np.float64)
    dbgbs=np.zeros(ntemp,dtype=np.float64)
    taublT=np.zeros(ntemp,dtype=np.float64)
    dblT=np.zeros(ntemp,dtype=np.float64)
    
    ###################################################################
    # pour definir les courbes contraintes/tailles de grain à différentes temperatures
    
    tau_NR=np.zeros(nd,dtype=np.float64)
    plt.figure(20)
    plt.title("Olivine Deformation mechanism map")
    # Loop on temperature
    ipl=0
    ifl=0
    for i in range(len(temp)):
        t=temp[i]
        if t>=ipllabel[ipl]: 
            ipl=ipl+1
            ifl=1
        # Loop on grain size
        for j in range(len(d)):
            gs=d[j]
            sigdis=(sr/Adis)**(1/ndis) * np.exp(Edis/ (ndis*Rgas*(t+273)))
            sigdiff=(sr/Adiff)**(1/ndiff) * gs**(mdiff/ndiff) * np.exp(Ediff/(ndiff*Rgas*(t+273)))
            siggbs=(sr/Agbs)**(1/ngbs) * gs**(mgbs/ngbs) * np.exp(Egbs / (ngbs*Rgas*(t+273)))
            siglowT=taulowT*(1-(-Rgas*(t+273)/ElowT * np.log(sr/AlowT))**(1/qlowT))**(1/plowT)
            sig=min(sigdis,sigdiff,siggbs)
            if t<TlowT: sig=min(sigdis,sigdiff,siggbs,siglowT)
    
            # NewtonRaphson Loop
            tau_NR[j] = optimize.newton(f,sig,args=(sr,gs,t),tol=1e-3, maxiter=10,fprime=None,fprime2=None)
            #Strain rates for each deformation mechanisms
            sr_dis,sr_diff,sr_gbs,sr_lowT=compute_sr(tau_NR[j],sr,gs,t)
            srmax=max(sr_dis,sr_diff,sr_gbs,sr_lowT)
            # Define the field limits
            if sr_dis < srmax :
                taubdis[i]=tau_NR[j]
                dbdis[i]=d[j]
            if sr_diff>=srmax :
                taubdiff[i]=tau_NR[j]
                dbdiff[i]=d[j]
            if sr_gbs >=srmax:
                taubgbs[i]=tau_NR[j]
                dbgbs[i]=d[j]
            if sr_lowT < srmax:
                taublT[i]=tau_NR[j]
                dblT[i]=d[j]
        # end loop on grain size
        # plotting stress/grain size curves, with label each 100°C
        if i%8==0:
           plt.plot(d,tau_NR,'k',linewidth=0.5)
        if ifl==1:
            plt.plot(d,tau_NR,linewidth=1.5,label='Temp='+str(int(t))+"°C")
            ifl=0
    # end loop on temperature
    
    # plotting limits between fields
    plt.plot(dbdiff,taubdiff,'--',linewidth=1.5,label='Equil Diff')
    #plt.plot(dbgbs,taubgbs,'--',linewidth=1.5,label='Equil GBS')
    plt.plot(dbdis,taubdis,'--',linewidth=1.5,label='Equil Dis')
    plt.plot(dblT,taublT,'--',linewidth=1.5,label='Equil Low T')
    plt.xlabel('Grain size (microns)')
    plt.ylabel('Stress (MPa)')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(bbox_to_anchor=(1.0, 1), loc='upper left', borderaxespad=0)
    #plt.legend()
    plt.grid(True)
    plt.axis([dmin,dmax, 1, 2e3])
    #plt.axis([np.log10(dmin),np.log10(dmax), np.log10(1), np.log10(2e3)])
    #plt.savefig('deformation_map_orig.png', bbox_inches='tight',dpi=200)
    plt.show()
#
#######

###### LOOP OVER HISTFILE
## define arrays


for i in range(len(files)):
    A=np.genfromtxt(files[i],skip_header=1)
    # istep,      \
    # total_time, \
    # totstrain12,  \
    # visc12,  \
    # sr12,  \
    # gs12,   \
    # totstrain2,  \
    # visc2, \
    # sr2,  \
    # gs2, \
    # sr,  \
    # temp))
    ttime=A[:,1]
    ttime*=1/Myr
    totalstrain12 = A[:,2]
    visc12 = A[:,3]
    sr12 = A[:,4]
    gs12 = A[:,5]
    totalstrain2 = A[:,6]
    visc2 = A[:,7]
    sr2 = A[:,8]
    gs2 = A[:,9]
    sr = A[:,10]
    temp = A[:,11]
    tau=A[:,7]
#
# compute weakening 
    weakening[i]=np.min(visc2)/np.max(visc2)
    strainloc[i]=np.max(sr2)/np.mean(sr)
# calculate  eps1 ans eps2 and tau
    viscmax=np.max(visc2)
    viscmin=np.min(visc2)
    for j in range(len(visc2)):
        if visc2[j] ==viscmax: jmax=j
        if visc2[j] == viscmin: jmin=j
        tau[j]=2*visc2[j]*sr2[j]
#
    eps1[i]=totalstrain2[jmax]
    eps2[i]=totalstrain2[jmin]
    weak=visc2/A[0,7]
    temperature[i]=np.mean(temp)
    if defmap==True: 
        plt.figure(20)
        plt.plot(gs2,tau*1e-6,marker='H',markersize = 2,label="T="+str(np.mean(temp)))
    plt.figure(1)
    plt.plot(totalstrain2,visc2,marker='H',markersize = 2,label="T="+str(np.mean(temp)))
    plt.xlabel('Strain')
    plt.ylabel('Viscosity')
    plt.figure(2)
    plt.plot(totalstrain2,weak,marker='H',markersize = 2,label="T="+str(np.mean(temp)))
    plt.xlabel('Strain')
    plt.ylabel('Weakening')
    #temp=float(re.search('pf=0.9_(.+?)', path).group(1))
    #print('per_incl',per_incl)
    # for history
plt.figure(20)
plt.savefig('deformation_map.png', bbox_inches='tight',dpi=200)
plt.figure(1)
plt.savefig('viscosity_strain.png', bbox_inches='tight',dpi=200)
plt.figure(2)
plt.savefig('weakening_strain.png', bbox_inches='tight',dpi=200)
plt.figure(3)
plt.plot(temperature,weakening,'o',markersize = 5)
plt.xlabel('Temperature')
plt.ylabel('Weakening')
plt.savefig('weakening_temperature.png', bbox_inches='tight',dpi=200)
plt.figure(4)
plt.plot(temperature,eps1,'o',markersize = 2)
plt.plot([np.min(temperature),np.max(temperature)],[np.mean(eps1),np.mean(eps1)],'k')
plt.plot(temperature,eps2,'o',markersize = 5)
plt.plot([np.min(temperature),np.max(temperature)],[np.mean(eps2),np.mean(eps2)],'k')
print('eps1 mean,',np.mean(eps1))
print('eps2 mean,',np.mean(eps2))
#plt.plot(temperature,eps2-eps1,'o',markersize = 5)
plt.xlabel('Temperature')
plt.ylabel('EPS1/EPS2')
plt.savefig('eps12_temperature.png', bbox_inches='tight',dpi=200)
plt.figure(5)
plt.plot(temperature,strainloc,'o',markersize = 5)
plt.xlabel('Temperature')
plt.ylabel('Strainlocalization')
plt.savefig('strainloc_temperature.png', bbox_inches='tight',dpi=200)




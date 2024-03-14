import numpy as np
from creep_parameters import *
from common import *
from rheology import *

###################################################################################################
# sr is total strain rate but when computing stresses one needs to remember that 
# each mechanism only sees half the strain rate, so eeq

sr=1e-15
eeq=sr/2

dinf_gbs_diff=np.zeros(600)
dinf_dis_diff=np.zeros(600)
tau_diff=np.zeros(600)
tau_gbs=np.zeros(600)
tau_dis=np.zeros(600)
dinf=np.zeros(600)
tau=np.zeros(600)

#----------------------
# interface diff gbs

for i in range(0,600):
    T=i+550+Tkelvin
    dinf_gbs_diff[i]=compute_dinf_gbs_diff(sr,T) 
    tau_diff[i]=(eeq/Adiff*np.exp(Ediff/Rgas/T)*dinf_gbs_diff[i]**mdiff)**(1/ndiff)
    tau_gbs[i] =(eeq/Agbs *np.exp(Egbs/Rgas/T) *dinf_gbs_diff[i]**mgbs) **(1/ngbs)
    #print(dinf_gbs_diff[i],tau_diff[i],tau_gbs[i],T)

np.savetxt('gbs_diff.ascii',np.array([dinf_gbs_diff,tau_diff,tau_gbs ]).T,header='# xx')

#----------------------
# interface dis gbs

for i in range(0,600):
    T=i+550+Tkelvin
    dinf_dis_diff[i]=compute_dinf_dis_diff(sr,T) 
    tau_diff[i]=(eeq/Adiff*np.exp(Ediff/Rgas/T)*dinf_dis_diff[i]**mdiff)**(1/ndiff)
    tau_dis[i]=(eeq/Adis*np.exp(Edis/Rgas/T))**(1/ndis)
    #print(dinf_dis_diff[i],tau_diff[i],tau_dis[i],T)

np.savetxt('dis_diff.ascii',np.array([dinf_dis_diff,tau_diff,tau_dis ]).T,header='# xx')

#----------------------
# unify interfaces

for i in range(0,600):
    T=i+550+Tkelvin
    if tau_gbs[i]<tau_dis[i]:
       tau[i]=tau_gbs[i]
       dinf[i]=dinf_gbs_diff[i]
    else:
       tau[i]=tau_dis[i]
       dinf[i]=dinf_dis_diff[i]
    #print(dinf[i],tau_diff[i],0,T)

np.savetxt('dinf.ascii',np.array([dinf,tau]).T,header='# xx')

#----------------------
# test function

for i in range(0,600):
    T=i+550+Tkelvin
    dinf[i],tau[i]=compute_dinf(sr,T)

np.savetxt('dinf2.ascii',np.array([dinf,tau]).T,header='# xx')


###################################################################################################

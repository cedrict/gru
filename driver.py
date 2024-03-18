import numpy as np
import gru
import time 
from common import *

print('******************************************')
print(' GGGGG  RRRR   U   U')
print(' G      R   R  U   U')
print(' G  GG  RRRR   U   U')
print(' G   G  R   R  U   U')
print(' GGGGG  R   R  UUUUU  0.1')
print('******************************************')

Lx=1*cm         # horizontal extent of the domain in m 
Ly=1*cm         # vertical extent of the domain 

sr=1e-15 # background strain rate

nely = 12  # number of elements in horizontal direction, pref multiple of 6
    
nmarker_per_dim=4

Tmin=700
Tmax=710
dtemp=50  
temp=np.arange(Tmin,Tmax,dtemp)
    
radius=Lx/8  # for mat 2 (mean value in a small circle)

gamma=2 #total deformation
    
CFL_nb=0.25

background_gs=1000 # initial grain size in microns

# kinetics of grain size reduction (0.75 works fine). Smaller-> slower grain size reduction
# this is epsilon_T parameter of eq.2 of prgu09 
egs=0.75 

rootfolder='./ABC/'

for i in range(len(temp)):
    start = time.time()
    gru.stonerheo(sr,temp[i],nely,egs,nmarker_per_dim,gamma,radius,Lx,Ly,CFL_nb,background_gs,rootfolder)
    end = time.time()
    print('>>>Temp=',temp[i],'C | total time=',end-start)

#gru.stonerheo(sr,Tmin,nely,egs)

print('******************************************')

import numpy as np
import gru
import time 

print('******************************************')
print(' GGGGG  RRRR   U   U')
print(' G      R   R  U   U')
print(' G  GG  RRRR   U   U')
print(' G   G  R   R  U   U')
print(' GGGGG  R   R  UUUUU  0.1')
print('******************************************')


sr=1e-15 # background strain rate

nely = 12  # number of elements in horizontal direction
    
nmarker_per_dim=6

Tmin=600
Tmax=700
dtemp=50  
temp=np.arange(Tmin,Tmax,dtemp)
    
radius=0.01/8  # for mat 2 (mean value in a small circle)

gamma=2 #total deformation

# kinetics of grain size reduction (0.75 works fine). Smaller-> slower grain size reduction
# this is epsilon_T parameter of eq.2 of prgu09 
egs=0.75 

for i in range(len(temp)):
    start = time.time()
    gru.stonerheo(sr,temp[i],nely,egs,nmarker_per_dim,gamma,radius)
    end = time.time()
    print('-->Temp=',temp[i],'C | time=',end-start)

#gru.stonerheo(sr,Tmin,nely,egs)

print('******************************************')

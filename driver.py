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
    
nmarker_per_dim=5

Tmin=600
Tmax=700
dtemp=100  
temp=np.arange(Tmin,Tmax,dtemp)

gamma=1 #total deformation

# kinetics of grain size reduction (0.75 works fine). Smaller-> slower grain size reduction
egs=0.75 

for i in range(len(temp)):
    start = time.time()
    gru.stonerheo(sr,temp[i],nely,egs,nmarker_per_dim,gamma)
    end = time.time()
    print('-->Temp=',temp[i],'C | time=',end-start)

#gru.stonerheo(sr,Tmin,nely,egs)

print('******************************************')

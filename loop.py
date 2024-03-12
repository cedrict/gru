import numpy as np
import gru
import time 

print('******************************************')
print(' GGGGG  RRRR   U   U')
print(' G      R   R  U   U')
print(' G  GG  RRRR   U   U')
print(' G   G  R   R  U   U')
print(' GGGGG  R   R  UUUUU  1.0')
print('******************************************')


sr=1e-15

nely = 16              # number of elements in horizontal direction
    
nmarker_per_dim=6

Tmin=500
Tmax=600
dtemp=10  
temp=np.arange(Tmin,Tmax,dtemp)

# kinetics of grain size reduction (0.75 works fine). Smaller-> slower grain size reduction
egs=0.75 

for i in range(len(temp)):
    start = time.time()
    gru.stonerheo(sr,temp[i],nely,egs,nmarker_per_dim)
    end = time.time()
    print('-->Temp=',temp[i],'C | time=',end-start)

#gru.stonerheo(sr,Tmin,nely,egs)

print('******************************************')

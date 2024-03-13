import numpy as np

###################################################################################################
#   Vspace=Q2     Pspace=Q1       
#
#  3----6----2   3---------2  
#  |    |    |   |         |  
#  |    |    |   |         |  
#  7----8----5   |         |  
#  |    |    |   |         |  
#  |    |    |   |         |  
#  0----4----1   0---------1  
###################################################################################################

rVnodes=[-1,+1,+1,-1, 0,+1, 0,-1,0]
sVnodes=[-1,-1,+1,+1,-1,0,+1,0,0]
    
def NNV(rq,sq):
    N0= 0.5*rq*(rq-1.) * 0.5*sq*(sq-1.)
    N1= 0.5*rq*(rq+1.) * 0.5*sq*(sq-1.)
    N2= 0.5*rq*(rq+1.) * 0.5*sq*(sq+1.)
    N3= 0.5*rq*(rq-1.) * 0.5*sq*(sq+1.)
    N4=     (1.-rq**2) * 0.5*sq*(sq-1.)
    N5= 0.5*rq*(rq+1.) *     (1.-sq**2)
    N6=     (1.-rq**2) * 0.5*sq*(sq+1.)
    N7= 0.5*rq*(rq-1.) *     (1.-sq**2)
    N8=     (1.-rq**2) *     (1.-sq**2)
    return np.array([N0,N1,N2,N3,N4,N5,N6,N7,N8],dtype=np.float64)
    
def dNNVdr(rq,sq):
    dNdr0= 0.5*(2.*rq-1.) * 0.5*sq*(sq-1)
    dNdr1= 0.5*(2.*rq+1.) * 0.5*sq*(sq-1)
    dNdr2= 0.5*(2.*rq+1.) * 0.5*sq*(sq+1)
    dNdr3= 0.5*(2.*rq-1.) * 0.5*sq*(sq+1)
    dNdr4=       (-2.*rq) * 0.5*sq*(sq-1)
    dNdr5= 0.5*(2.*rq+1.) *    (1.-sq**2)
    dNdr6=       (-2.*rq) * 0.5*sq*(sq+1)
    dNdr7= 0.5*(2.*rq-1.) *    (1.-sq**2)
    dNdr8=       (-2.*rq) *    (1.-sq**2)
    return np.array([dNdr0,dNdr1,dNdr2,dNdr3,dNdr4,dNdr5,dNdr6,dNdr7,dNdr8],dtype=np.float64)
    
def dNNVds(rq,sq):
    dNds0= 0.5*rq*(rq-1.) * 0.5*(2.*sq-1.)
    dNds1= 0.5*rq*(rq+1.) * 0.5*(2.*sq-1.)
    dNds2= 0.5*rq*(rq+1.) * 0.5*(2.*sq+1.)
    dNds3= 0.5*rq*(rq-1.) * 0.5*(2.*sq+1.)
    dNds4=     (1.-rq**2) * 0.5*(2.*sq-1.)
    dNds5= 0.5*rq*(rq+1.) *       (-2.*sq)
    dNds6=     (1.-rq**2) * 0.5*(2.*sq+1.)
    dNds7= 0.5*rq*(rq-1.) *       (-2.*sq)
    dNds8=     (1.-rq**2) *       (-2.*sq)
    return np.array([dNds0,dNds1,dNds2,dNds3,dNds4,dNds5,dNds6,dNds7,dNds8],dtype=np.float64)
    
def NNP(rq,sq):
    N0=0.25*(1-rq)*(1-sq)
    N1=0.25*(1+rq)*(1-sq)
    N2=0.25*(1+rq)*(1+sq)
    N3=0.25*(1-rq)*(1+sq)
    return np.array([N0,N1,N2,N3],dtype=np.float64)
    
###################################################################################################

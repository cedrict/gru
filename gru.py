import numpy as np
import scipy.sparse as sps
from scipy.sparse import lil_matrix 
import time 
import os
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random
from FEbasis import *
from common import *
from rheology import *
from write_history import *
    
###################################################################################################

def stonerheo(background_sr,tempdegC,nely,egs,nmarker_per_dim,gamma,radius,Lx,Ly,CFL_nb,background_gs,rootfolder):

    #max number of time steps
    nstep=100

    #usefull parameters
    background_T=tempdegC+Tkelvin 

    #markers parameters
    avrg=3

    #nonlinear iterations parameters
    niter=10
    tol=1e-3

    # processing input    
    v0=background_sr*Ly                  # bc velocity so that shear strain rate is the imposed strain rate sr
    nelx = int(nely*Lx/Ly)               # number of elements in vertical direction
    permat2=np.pi*radius**2/(Lx*Ly)*100  # percentage of disc wrt whole domain
    tfinal=gamma/background_sr           # total time to reach gamma  

    ###############################################################################################
    
    output_folder=rootfolder+'GrainSize-T-'+str(int(tempdegC))+'Kinetics-'+str(egs)+'/'
    if not os.path.isdir(rootfolder): 
       os.mkdir(rootfolder)
    if not os.path.isdir(output_folder): 
       os.mkdir(output_folder)  
    convfile=open(output_folder+'conv.ascii',"w")
    logfile=open(output_folder+'log.ascii',"w")
    swarmgsfile=open(output_folder+'swarm_gs.ascii',"w")
    histfile=rootfolder+'Hist-T-'+str(int(tempdegC))+'-Sr-'+str(-int(np.log10(background_sr)))+'-Kinetics-'+str(egs)+'-PercMat2-'+str(int(permat2))

    print('  |results in:'+output_folder)
    print('  |'+histfile)

    ###############################################################################################
    
    mV=9     # number of velocity nodes making up an element
    mP=4     # number of pressure nodes making up an element
    ndofV=2  # number of velocity degrees of freedom per node
    ndofP=1  # number of pressure degrees of freedom 
        
    nnx=2*nelx+1  # number of elements, x direction
    nny=2*nely+1  # number of elements, y direction
    
    NV=nnx*nny           # number of V nodes
    nel=nelx*nely        # number of elements, total
    NP=(nelx+1)*(nely+1) # number of P nodes
    
    NfemV=NV*ndofV   # number of velocity dofs
    NfemP=NP*ndofP   # number of pressure dofs
    Nfem=NfemV+NfemP # total number of dofs
    
    nqperdim=3
    qcoords=[-np.sqrt(3./5.),0.,np.sqrt(3./5.)]
    qweights=[5./9.,8./9.,5./9.]
    
    hx=Lx/nelx # size of element
    hy=Ly/nely
    
    nmarker_per_element=nmarker_per_dim**2
    nmarker=nel*nmarker_per_element
    
    #################################################################
    #################################################################
    logfile.write("nelx:"+str(nelx)+"\n")
    logfile.write("nely:"+str(nely)+"\n")
    logfile.write("nel: "+str(nel)+"\n")
    logfile.write("nnx: "+str(nnx)+"\n")
    logfile.write("nny: "+str(nny)+"\n")
    logfile.write("NV:  "+str(NV)+"\n")
    logfile.write("tfinal:"+str(tfinal/year/1e6)+"Myr \n")
    logfile.write("------------------------------\n")
    
    eta_ref=1e20 # purely numerical parameter - do not change

    time_localise_markers=0
    time_advect_markers=0
    time_build_matrix=0
    time_solve_system=0
    time_compute_viscosity=0
    time_compute_sr=0
    
    #################################################################
    # grid point setup
    #################################################################
    start = time.time()
    
    xV=np.empty(NV,dtype=np.float64)  # x coordinates
    yV=np.empty(NV,dtype=np.float64)  # y coordinates
    
    counter = 0
    for j in range(0,nny):
        for i in range(0,nnx):
            xV[counter]=i*hx/2.
            yV[counter]=j*hy/2.
            counter += 1
        #end for
    #end for
    
    iconV=np.zeros((mV,nel),dtype=np.int32)
    counter = 0
    for j in range(0,nely):
        for i in range(0,nelx):
            iconV[0,counter]=(i)*2+1+(j)*2*nnx -1
            iconV[1,counter]=(i)*2+3+(j)*2*nnx -1
            iconV[2,counter]=(i)*2+3+(j)*2*nnx+nnx*2 -1
            iconV[3,counter]=(i)*2+1+(j)*2*nnx+nnx*2 -1
            iconV[4,counter]=(i)*2+2+(j)*2*nnx -1
            iconV[5,counter]=(i)*2+3+(j)*2*nnx+nnx -1
            iconV[6,counter]=(i)*2+2+(j)*2*nnx+nnx*2 -1
            iconV[7,counter]=(i)*2+1+(j)*2*nnx+nnx -1
            iconV[8,counter]=(i)*2+2+(j)*2*nnx+nnx -1
            counter += 1
        #end for
    #end for

    xc=np.empty(nel,dtype=np.float64)  # x coordinates
    yc=np.empty(nel,dtype=np.float64)  # y coordinates

    for iel in range(0,nel): 
        xc[iel]=0.5*(xV[iconV[0,iel]]+xV[iconV[2,iel]])
        yc[iel]=0.5*(yV[iconV[0,iel]]+yV[iconV[2,iel]])
    
    logfile.write("velocity grid points: %.3f s \n" % (time.time() - start))
    
    #################################################################
    # pressure connectivity array
    #################################################################
    start = time.time()
    
    xP=np.zeros(NP,dtype=np.float64)     # x coordinates
    yP=np.zeros(NP,dtype=np.float64)     # y coordinates
    iconP=np.zeros((mP,nel),dtype=np.int32)
    
    counter = 0
    for j in range(0,nely):
        for i in range(0,nelx):
            iconP[0,counter]=i+j*(nelx+1)
            iconP[1,counter]=i+1+j*(nelx+1)
            iconP[2,counter]=i+1+(j+1)*(nelx+1)
            iconP[3,counter]=i+(j+1)*(nelx+1)
            counter += 1
        #end for
    #end for
    
    counter = 0
    for j in range(0, nely+1):
        for i in range(0, nelx+1):
            xP[counter]=i*Lx/float(nelx)
            yP[counter]=j*Ly/float(nely)
            counter += 1
        #end for
    #end for
    
    logfile.write("pressure connectivity & nodes: %.3f s \n" % (time.time() - start))
    
    #################################################################
    # compute area of elements
    # This is a good test because it uses the quadrature points and 
    # weights as well as the shape functions. If any area comes out
    # negative or zero, or if the sum does not equal to the area of the 
    # whole domain then there is a major problem which needs to 
    # be addressed before FE are set into motion.
    #################################################################
    start = time.time()
    
    area=np.zeros(nel,dtype=np.float64) 
    
    for iel in range(0,nel):
        for iq in range(0,nqperdim):
            for jq in range(0,nqperdim):
                rq=qcoords[iq]
                sq=qcoords[jq]
                weightq=qweights[iq]*qweights[jq]
                NNNV=NNV(rq,sq)
                dNNNVdr=dNNVdr(rq,sq)
                dNNNVds=dNNVds(rq,sq)
                jcb=np.zeros((2,2),dtype=np.float64)
                for k in range(0,mV):
                    jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                    jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                    jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                    jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
                jcob = np.linalg.det(jcb)
                area[iel]+=jcob*weightq
            if area[iel]<0: 
               for k in range(0,mV):
                   print (xV[iconV[k,iel]],yV[iconV[k,iel]])
            #end for
        #end for
    #end for
    
    logfile.write("     -> area (m,M) %.6e %.6e \n" %(np.min(area),np.max(area)))
    logfile.write("     -> total area meas %.8e \n" %(area.sum()))
    logfile.write("     -> total area anal %.8e \n" %(Lx*Ly))
    
    logfile.write("compute elements areas: %.3f s \n" % (time.time() - start))
    
    #################################################################
    # define boundary conditions
    #################################################################
    start = time.time()
    
    bc_fix=np.zeros(NfemV,dtype=bool) # boundary condition, yes/no  
    bc_val=np.zeros(NfemV,dtype=np.float64) # boundary condition, value
    
    for i in range(0, NV):
        #left boundary 
        if xV[i]<eps:
           bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0 # vy
        #right boundary 
        if xV[i]/Lx>1-eps:
           bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0 # vy
        #bottom boundary 
        if yV[i]/Ly<eps:
           bc_fix[i*ndofV+0] = True ; bc_val[i*ndofV  ] = -v0 # vx
           bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0   # vy
        #top boundary 
        if yV[i]/Ly>1-eps:
           bc_fix[i*ndofV+0] = True ; bc_val[i*ndofV  ] = +v0 # vx
           bc_fix[i*ndofV+1] = True ; bc_val[i*ndofV+1] = 0   # vy
    #end for
    
    logfile.write("setup: boundary conditions: %.3f s \n" % (time.time() - start))
    
    #################################################################
    # swarm (=all the particles) setup
    #################################################################
    start = time.time()
    
    swarm_x=np.zeros(nmarker,dtype=np.float64)                  # x coordinates   
    swarm_y=np.zeros(nmarker,dtype=np.float64)                  # y coordinates 
    swarm_mat=np.zeros(nmarker,dtype=np.int8)                   # type of material 
    swarm_paint=np.zeros(nmarker,dtype=np.float64)              # paint
    swarm_r=np.zeros(nmarker,dtype=np.float64)                  # reduced coordinates r
    swarm_s=np.zeros(nmarker,dtype=np.float64)                  # reduced coordinates s
    swarm_u=np.zeros(nmarker,dtype=np.float64)                  # velocity x
    swarm_v=np.zeros(nmarker,dtype=np.float64)                  # velocity y
    swarm_exx=np.zeros(nmarker,dtype=np.float64)                # strain rate xx
    swarm_exy=np.zeros(nmarker,dtype=np.float64)                # strain rate yy
    swarm_eyy=np.zeros(nmarker,dtype=np.float64)                # strain rate xy
    swarm_ee=np.zeros(nmarker,dtype=np.float64)                 # effective strain rate
    swarm_total_strainxx=np.zeros(nmarker,dtype=np.float64)     # strain xx
    swarm_total_strainxy=np.zeros(nmarker,dtype=np.float64)     # strain yy
    swarm_total_strainyy=np.zeros(nmarker,dtype=np.float64)     # strain xy
    swarm_total_strain_eff=np.zeros(nmarker,dtype=np.float64)   # effective strain 
    swarm_gs=np.zeros(nmarker,dtype=np.float64)                 # grain size 
    swarm_tauxx=np.zeros(nmarker,dtype=np.float64)              # dev stress xx
    swarm_tauxy=np.zeros(nmarker,dtype=np.float64)              # dev stress yy
    swarm_tauyy=np.zeros(nmarker,dtype=np.float64)              # dev stress xy
    swarm_tau_eff=np.zeros(nmarker,dtype=np.float64)            # effective dev stress
    swarm_iel=np.zeros(nmarker,dtype=np.int32)                  # element identity
    swarm_eta=np.zeros(nmarker,dtype=np.float64)                # viscosity
    swarm_tau_angle=np.zeros(nmarker,dtype=np.float64)          # principal angle  
    swarm_sigmaxx=np.zeros(nmarker,dtype=np.float64)            # full stress xx
    swarm_sigmaxy=np.zeros(nmarker,dtype=np.float64)            # full stress yy
    swarm_sigmayy=np.zeros(nmarker,dtype=np.float64)            # full stress xy
    swarm_sigma_angle=np.zeros(nmarker,dtype=np.float64)        # principal angle  
    swarm_sigma1=np.zeros(nmarker,dtype=np.float64)             # principal stress
    swarm_sigma2=np.zeros(nmarker,dtype=np.float64)             # principal stress 
    swarm_sr_dis=np.zeros(nmarker,dtype=np.float64)             # strain rate 
    swarm_sr_diff=np.zeros(nmarker,dtype=np.float64)            # strain rate 
    swarm_sr_gbs=np.zeros(nmarker,dtype=np.float64)             # strain rate 
    swarm_sr_lowT=np.zeros(nmarker,dtype=np.float64)            # strain rate 
    swarm_dinf=np.zeros(nmarker,dtype=np.float64)               # d_inf
    swarm_tauNR=np.zeros(nmarker,dtype=np.float64)              # 
    
    counter=0
    for iel in range(0,nel):
        x1=xV[iconV[0,iel]] ; y1=yV[iconV[0,iel]]
        x2=xV[iconV[1,iel]] ; y2=yV[iconV[1,iel]]
        x3=xV[iconV[2,iel]] ; y3=yV[iconV[2,iel]]
        x4=xV[iconV[3,iel]] ; y4=yV[iconV[3,iel]]
        for j in range(0,nmarker_per_dim):
            for i in range(0,nmarker_per_dim):
                r=-1.+i*2./nmarker_per_dim + 1./nmarker_per_dim
                s=-1.+j*2./nmarker_per_dim + 1./nmarker_per_dim
                swarm_r[counter]=r
                swarm_s[counter]=s
                N1=0.25*(1-r)*(1-s)
                N2=0.25*(1+r)*(1-s)
                N3=0.25*(1+r)*(1+s)
                N4=0.25*(1-r)*(1+s)
                swarm_x[counter]=N1*x1+N2*x2+N3*x3+N4*x4
                swarm_y[counter]=N1*y1+N2*y2+N3*y3+N4*y4
                counter+=1
            #end for 
        #end for 
    #end for 
    
    logfile.write("swarm setup: %.3f s \n" % (time.time() - start))
    
    #################################################################
    # assign material id and gs to markers 
    #################################################################
    start = time.time()
    
    for im in range (0,nmarker):
        swarm_mat[im]=1
        swarm_gs[im]=random.gauss(background_gs,0.1*background_gs)  # initial grain size 
        #swarm_gs[im]=background_gs
        xxi=swarm_x[im]-Lx/2
        yyi=swarm_y[im]-Ly/2
        if xxi**2/radius**2+yyi**2/radius**2<1: swarm_mat[im]=2
        if swarm_y[im]<1/6*Ly or swarm_y[im]>5/6*Ly: 
            swarm_gs[im]=background_gs
            swarm_mat[im]=3
    
    logfile.write("assigning mat to swarm: %.3f s\n" % (time.time() - start))
    
    #################################################################
    # paint markers 
    #################################################################
    start = time.time()
    
    for im in range (0,nmarker):
        for i in range(0,5):
            if abs(swarm_x[im]-i*Lx/4)<hx/4:
               swarm_paint[im]=1
        #end for 
        for i in range(0,5):
            if abs(swarm_y[im]-i*Ly/4)<hy/4:
               swarm_paint[im]=1
        #end for 
    #end for 
    
    logfile.write("paint swarm: %.3f s\n" % (time.time() - start))
    
    ###################################################################################################
    ###################################################################################################
    # time stepping loop 
    ###################################################################################################
    ###################################################################################################
    
    total_time=0
        
    umem=np.zeros(NV,dtype=np.float64)      
    vmem=np.zeros(NV,dtype=np.float64)    
    u   =np.zeros(NV,dtype=np.float64)   
    v   =np.zeros(NV,dtype=np.float64)    
    exx =np.zeros(NV,dtype=np.float64)  
    eyy =np.zeros(NV,dtype=np.float64)  
    exy =np.zeros(NV,dtype=np.float64) ; exy[:]=background_sr 
    
    for istep in range(0,nstep):
    
        logfile.write('||******************************************************||\n')
        logfile.write('||    istep= %i \n' %istep)
        logfile.write('||******************************************************||\n')
    
        #################################################################
        # localise markers 
        #################################################################
        start = time.time()
    
        nmarker_in_element=np.zeros(nel,dtype=np.int16)
        list_of_markers_in_element=np.zeros((2*nmarker_per_element,nel),dtype=np.int32)
        for im in range(0,nmarker):
            ielx=int(swarm_x[im]/Lx*nelx)
            iely=int(swarm_y[im]/Ly*nely)
            iel=nelx*(iely)+ielx
            list_of_markers_in_element[nmarker_in_element[iel],iel]=im
            nmarker_in_element[iel]+=1
            swarm_iel[im]=iel
        #end for
    
        logfile.write("     localise markers: %.3f s \n" % (time.time() - start))

        time_localise_markers+=time.time() - start
    
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # non linear iterations
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
        for iter in range(0,niter):
    
            logfile.write('     ----------------------------------\n')
            logfile.write('     ----- iteration------'+str(iter)+'----------\n')
            logfile.write('     ----------------------------------\n')
    
            #################################################################
            # compute elemental averagings 
            #################################################################
            start = time.time()
            eta_elemental=np.zeros(nel,dtype=np.float64)
            eta_nodal=np.zeros(NP,dtype=np.float64)
            nodal_counter=np.zeros(NP,dtype=np.float64)
            for im in range(0,nmarker):
                iel=swarm_iel[im]
                N1=0.25*(1-swarm_r[im])*(1-swarm_s[im])
                N2=0.25*(1+swarm_r[im])*(1-swarm_s[im])
                N3=0.25*(1+swarm_r[im])*(1+swarm_s[im])
                N4=0.25*(1-swarm_r[im])*(1+swarm_s[im])
                nodal_counter[iconP[0,iel]]+=N1
                nodal_counter[iconP[1,iel]]+=N2
                nodal_counter[iconP[2,iel]]+=N3
                nodal_counter[iconP[3,iel]]+=N4
                #compute visc on marker
                NNNV[0:mV]=NNV(swarm_r[im],swarm_s[im])
                swarm_exx[im]=sum(NNNV[0:mV]*exx[iconV[0:mV,iel]])
                swarm_eyy[im]=sum(NNNV[0:mV]*eyy[iconV[0:mV,iel]])
                swarm_exy[im]=sum(NNNV[0:mV]*exy[iconV[0:mV,iel]])
                swarm_ee[im]=np.sqrt(0.5*(swarm_exx[im]**2+swarm_eyy[im]**2+2*swarm_exy[im]**2) ) 
                swarm_eta[im],swarm_tauNR[im],swarm_sr_dis[im],swarm_sr_diff[im],swarm_sr_gbs[im],swarm_sr_lowT[im]\
                =viscosity(swarm_ee[im],background_T,swarm_mat[im],swarm_gs[im])
    
                if abs(avrg)==1 : # arithmetic
                   eta_elemental[iel]     +=swarm_eta[im]
                   eta_nodal[iconP[0,iel]]+=swarm_eta[im]*N1
                   eta_nodal[iconP[1,iel]]+=swarm_eta[im]*N2
                   eta_nodal[iconP[2,iel]]+=swarm_eta[im]*N3
                   eta_nodal[iconP[3,iel]]+=swarm_eta[im]*N4
                if abs(avrg)==2: # geometric
                   eta_elemental[iel]     +=np.log10(swarm_eta[im])
                   eta_nodal[iconP[0,iel]]+=np.log10(swarm_eta[im])*N1
                   eta_nodal[iconP[1,iel]]+=np.log10(swarm_eta[im])*N2
                   eta_nodal[iconP[2,iel]]+=np.log10(swarm_eta[im])*N3
                   eta_nodal[iconP[3,iel]]+=np.log10(swarm_eta[im])*N4
                if abs(avrg)==3: # harmonic
                   eta_elemental[iel]     +=1/swarm_eta[im]
                   eta_nodal[iconP[0,iel]]+=1/swarm_eta[im]*N1
                   eta_nodal[iconP[1,iel]]+=1/swarm_eta[im]*N2
                   eta_nodal[iconP[2,iel]]+=1/swarm_eta[im]*N3
                   eta_nodal[iconP[3,iel]]+=1/swarm_eta[im]*N4
            #end for
            if abs(avrg)==1:
               eta_nodal/=nodal_counter
               eta_elemental[:]/=nmarker_in_element[:]
            if abs(avrg)==2:
               eta_nodal[:]=10.**(eta_nodal[:]/nodal_counter[:])
               eta_elemental[:]=10.**(eta_elemental[:]/nmarker_in_element[:])
            if abs(avrg)==3:
               eta_nodal[:]=nodal_counter[:]/eta_nodal[:]
               eta_elemental[:]=nmarker_in_element[:]/eta_elemental[:]
    
            logfile.write("          -> nmarker_in_elt(m,M) %.5e %.5e \n" %(np.min(nmarker_in_element),np.max(nmarker_in_element)))
            logfile.write("          -> eta_elemental (m,M) %.5e %.5e \n" %(np.min(eta_elemental),np.max(eta_elemental)))
            logfile.write("          -> eta_nodal     (m,M) %.5e %.5e \n" %(np.min(eta_nodal),np.max(eta_nodal)))
    
            logfile.write("     markers onto grid: %.3f s\n" % (time.time() - start))

            time_compute_viscosity+=time.time() - start
    
            #################################################################
            # build FE matrix
            # [ K G ][u]=[f]
            # [GT 0 ][p] [h]
            #################################################################
            start = time.time()
    
            A_sparse = lil_matrix((Nfem,Nfem),dtype=np.float64)
    
            f_rhs   = np.zeros(NfemV,dtype=np.float64)         # right hand side f 
            h_rhs   = np.zeros(NfemP,dtype=np.float64)         # right hand side h 
            b_mat   = np.zeros((3,ndofV*mV),dtype=np.float64)  # gradient matrix B 
            N_mat   = np.zeros((3,ndofP*mP),dtype=np.float64)  # matrix  
            dNNNVdx = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
            dNNNVdy = np.zeros(mV,dtype=np.float64)            # shape functions derivatives
            c_mat   = np.array([[2,0,0],[0,2,0],[0,0,1]],dtype=np.float64) 

            jcb=np.zeros((2,2),dtype=np.float64)
            jcbi=np.zeros((2,2),dtype=np.float64)
            jcb[0,0]=hx/2
            jcb[1,1]=hy/2
            jcob=jcb[0,0]*jcb[1,1]
            jcbi[0,0]=2/hx
            jcbi[1,1]=2/hy
    
            for iel in range(0,nel):
    
                f_el =np.zeros((mV*ndofV),dtype=np.float64)
                K_el =np.zeros((mV*ndofV,mV*ndofV),dtype=np.float64)
                G_el=np.zeros((mV*ndofV,mP*ndofP),dtype=np.float64)
                h_el=np.zeros((mP*ndofP),dtype=np.float64)
    
                for iq in range(0,nqperdim):
                    for jq in range(0,nqperdim):
    
                        # position & weight of quad. point
                        rq=qcoords[iq]
                        sq=qcoords[jq]
                        weightq=qweights[iq]*qweights[jq]
    
                        NNNV=NNV(rq,sq)
                        dNNNVdr=dNNVdr(rq,sq)
                        dNNNVds=dNNVds(rq,sq)
                        NNNP=NNP(rq,sq)
    
                        # calculate jacobian matrix
                        # since elts are rectangles I can precompute this
                        #jcb=np.zeros((2,2),dtype=np.float64)
                        #for k in range(0,mV):
                        #    jcb[0,0] += dNNNVdr[k]*xV[iconV[k,iel]]
                        #    jcb[0,1] += dNNNVdr[k]*yV[iconV[k,iel]]
                        #    jcb[1,0] += dNNNVds[k]*xV[iconV[k,iel]]
                        #    jcb[1,1] += dNNNVds[k]*yV[iconV[k,iel]]
                        #jcob = np.linalg.det(jcb)
                        #jcbi = np.linalg.inv(jcb)
    
                        # compute dNdx & dNdy
                        for k in range(0,mV):
                            dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                            dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
    
                        # construct b_mat matrix
                        for i in range(0,mV):
                            b_mat[0:3, 2*i:2*i+2] = [[dNNNVdx[i],0.       ],
                                                     [0.        ,dNNNVdy[i]],
                                                     [dNNNVdy[i],dNNNVdx[i]]]
    
                        # compute elemental a_mat matrix
                        K_el+=b_mat.T.dot(c_mat.dot(b_mat))*eta_elemental[iel]*weightq*jcob
    
                        # compute elemental rhs vector
                        #for i in range(0,mV):
                        #    f_el[ndofV*i  ]+=NNNV[i]*jcob*weightq*gx*density(xq,yq)
                        #    f_el[ndofV*i+1]+=NNNV[i]*jcob*weightq*gy*density(xq,yq)
    
                        for i in range(0,mP):
                            N_mat[0,i]=NNNP[i]
                            N_mat[1,i]=NNNP[i]
                            N_mat[2,i]=0.
    
                        G_el-=b_mat.T.dot(N_mat)*weightq*jcob
    
                    # end for jq
                # end for iq
    
                # impose b.c. 
                for k1 in range(0,mV):
                    for i1 in range(0,ndofV):
                        ikk=ndofV*k1          +i1
                        m1 =ndofV*iconV[k1,iel]+i1
                        if bc_fix[m1]:
                           K_ref=K_el[ikk,ikk] 
                           for jkk in range(0,mV*ndofV):
                               f_el[jkk]-=K_el[jkk,ikk]*bc_val[m1]
                               K_el[ikk,jkk]=0
                               K_el[jkk,ikk]=0
                           K_el[ikk,ikk]=K_ref
                           f_el[ikk]=K_ref*bc_val[m1]
                           h_el[:]-=G_el[ikk,:]*bc_val[m1]
                           G_el[ikk,:]=0
                    #end for
                #end for
    
                G_el*=eta_ref/Ly
                h_el*=eta_ref/Ly
    
                # assemble matrix K_mat and right hand side rhs
                for k1 in range(0,mV):
                    for i1 in range(0,ndofV):
                        ikk=ndofV*k1          +i1
                        m1 =ndofV*iconV[k1,iel]+i1
                        for k2 in range(0,mV):
                            for i2 in range(0,ndofV):
                                jkk=ndofV*k2          +i2
                                m2 =ndofV*iconV[k2,iel]+i2
                                A_sparse[m1,m2] += K_el[ikk,jkk]
                            #end for
                        #end for
                        for k2 in range(0,mP):
                            jkk=k2
                            m2 =iconP[k2,iel]
                            A_sparse[m1,NfemV+m2]+=G_el[ikk,jkk]
                            A_sparse[NfemV+m2,m1]+=G_el[ikk,jkk]
                        #end for
                        f_rhs[m1]+=f_el[ikk]
                    #end for
                #end for
                for k2 in range(0,mP):
                    m2=iconP[k2,iel]
                    h_rhs[m2]+=h_el[k2]
                #end for
            #end for iel
    
            logfile.write("     build FE matrix: %.3f s \n" % (time.time() - start))

            time_build_matrix+=time.time() - start
    
            ######################################################################
            # solve system 
            ######################################################################
            start = time.time()
    
            rhs=np.zeros(Nfem,dtype=np.float64)   # right hand side of Ax=b
    
            sparse_matrix=A_sparse.tocsr()
    
            rhs[0:NfemV]=f_rhs
            rhs[NfemV:Nfem]=h_rhs
    
            sol=sps.linalg.spsolve(sparse_matrix,rhs)
    
            logfile.write("     solve time: %.3f s \n" % (time.time() - start))

            time_solve_system+=time.time() - start
    
            ######################################################################
            # put solution into separate x,y velocity arrays
            ######################################################################
            start = time.time()
    
            u,v=np.reshape(sol[0:NfemV],(NV,2)).T
            p=sol[NfemV:Nfem]*(eta_ref/Ly)
    
            logfile.write("          -> u (m,M) %.4e %.4e (cm/year)\n" %(np.min(u)/cm*year,np.max(u)/cm*year))
            logfile.write("          -> v (m,M) %.4e %.4e (cm/year)\n" %(np.min(v)/cm*year,np.max(v)/cm*year))
            logfile.write("          -> p (m,M) %.4e %.4e (Pa)     \n" %(np.min(p),np.max(p)))
    
            #np.savetxt('velocity.ascii',np.array([xV,yV,u,v]).T,header='# x,y,u,v')
            #np.savetxt('pressure_bef.ascii',np.array([xP,yP,p]).T,header='# x,y,p')
    
            logfile.write("     split vel into u,v: %.3f s \n" % (time.time() - start))
    
            ######################################################################
            #normalise pressure
            ######################################################################
            start = time.time()
    
            avrg_p=0
            for iel in range(0,nel):
                for iq in range(0,nqperdim):
                    for jq in range(0,nqperdim):
                        rq=qcoords[iq]
                        sq=qcoords[jq]
                        weightq=qweights[iq]*qweights[jq]
                        NNNP=NNP(rq,sq)
                        pq=NNNP.dot(p[iconP[0:mP,iel]])
                        avrg_p+=pq*jcob*weightq

            avrg_p/=(Lx*Ly)
 
            logfile.write('          -> avrg_p='+str(avrg_p)+'\n')
    
            p-=avrg_p
    
            logfile.write("          -> p (m,M) %.4e %.4e (Pa)   \n" %(np.min(p),np.max(p)))
                
            logfile.write("     pressure normalisation: %.3f s \n" % (time.time() - start))
    
            ######################################################################
            # compute nodal strainrate 
            ######################################################################
            start = time.time()
     
            exx = np.zeros(NV,dtype=np.float64)  
            eyy = np.zeros(NV,dtype=np.float64)  
            exy = np.zeros(NV,dtype=np.float64)  
            ccc = np.zeros(NV,dtype=np.float64)  
     
            for iel in range(0,nel):
                for k in range(0,mV):
                    rq = rVnodes[k]
                    sq = sVnodes[k]
                    inode=iconV[k,iel]
                    NNNV=NNV(rq,sq)
                    dNNNVdr=dNNVdr(rq,sq)
                    dNNNVds=dNNVds(rq,sq)
                    #jcb=np.zeros((2,2),dtype=np.float64)
                    #for k in range(0,mV):
                    #    jcb[0,0]+=dNNNVdr[k]*xV[iconV[k,iel]]
                    #    jcb[0,1]+=dNNNVdr[k]*yV[iconV[k,iel]]
                    #    jcb[1,0]+=dNNNVds[k]*xV[iconV[k,iel]]
                    #    jcb[1,1]+=dNNNVds[k]*yV[iconV[k,iel]]
                    #jcbi=np.linalg.inv(jcb)
                    #for k in range(0,mV):
                    #    dNNNVdx[k]=jcbi[0,0]*dNNNVdr[k]+jcbi[0,1]*dNNNVds[k]
                    #    dNNNVdy[k]=jcbi[1,0]*dNNNVdr[k]+jcbi[1,1]*dNNNVds[k]
                    #for k in range(0,mV):
                    dNNNVdx[:]=jcbi[0,0]*dNNNVdr[:]
                    dNNNVdy[:]=jcbi[1,1]*dNNNVds[:]
                    ccc[inode]+=1
                    exx[inode]+=dNNNVdx.dot(u[iconV[:,iel]])
                    eyy[inode]+=dNNNVdy.dot(v[iconV[:,iel]])
                    exy[inode]+=dNNNVdx.dot(v[iconV[:,iel]])*0.5+dNNNVdy.dot(u[iconV[:,iel]])*0.5
                #end for
            #end for
            exx[:]/=ccc[:]
            eyy[:]/=ccc[:]
            exy[:]/=ccc[:]
     
            logfile.write("          -> exx (m,M) %.4e %.4e \n" %(np.min(exx),np.max(exx)))
            logfile.write("          -> eyy (m,M) %.4e %.4e \n" %(np.min(eyy),np.max(eyy)))
            logfile.write("          -> exy (m,M) %.4e %.4e \n" %(np.min(exy),np.max(exy)))
     
            logfile.write("     compute strain rate: %.3f s \n" % (time.time() - start))

            time_compute_sr+=time.time() - start
    
            ######################################################################
            # convergence criterion is based on difference between two consecutively
            # obtained velocity fields, normalised by the boundary condition velocity
    
            chi_u=LA.norm(u-umem,2)/v0 # vx convergence indicator
            chi_v=LA.norm(v-vmem,2)/v0 # vy convergence indicator
    
            logfile.write('          -> convergence u,v: %.3e %.3e | tol= %.2e \n' %(chi_u,chi_v,tol))
    
            convfile.write("%f %10e %10e %10e\n" %(istep+iter/200,chi_u,chi_v,tol))
            convfile.flush()
    
            umem[:]=u[:]
            vmem[:]=v[:]
    
            if chi_u<tol and chi_v<tol:
               logfile.write('     ***converged*** \n')
               break
    
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #end for iter
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
        logfile.write('     ----------------------------------\n')
        logfile.write('     end of nl iteration \n')
        logfile.write('     ----------------------------------\n')
    
        ######################################################################
        # compute timestep using CFL condition 
        ######################################################################
    
        dt=CFL_nb*min(hx,hy)/max(max(abs(u)),max(abs(v)))
    
        total_time+=dt
    
        logfile.write("     -> dt= %.3e yr \n" %(dt/year))
        logfile.write("     -> time= %.3e yr \n" %(total_time/year))
    
        ######################################################################
        # advect markers 
        # a simple one-step Euler method is used so timesteps should
        # be kept to rather low values, i.e. CFL_nb<=0.25
        # Periodic boundary conditions are implemented on the markers.
        ######################################################################
        start = time.time()
    
        for im in range(0,nmarker):
            ielx=int(swarm_x[im]/Lx*nelx)
            iely=int(swarm_y[im]/Ly*nely)
            iel=nelx*iely+ielx
            swarm_r[im]=-1.+2*(swarm_x[im]-xV[iconV[0,iel]])/hx
            swarm_s[im]=-1.+2*(swarm_y[im]-yV[iconV[0,iel]])/hy
            NNNV=NNV(swarm_r[im],swarm_s[im])
            swarm_u[im]=sum(NNNV[0:mV]*u[iconV[0:mV,iel]])
            swarm_v[im]=sum(NNNV[0:mV]*v[iconV[0:mV,iel]])
            #swarm_exx[im]=sum(NNNV[0:mV]*exx[iconV[0:mV,iel]])
            #swarm_eyy[im]=sum(NNNV[0:mV]*eyy[iconV[0:mV,iel]])
            #swarm_exy[im]=sum(NNNV[0:mV]*exy[iconV[0:mV,iel]])
            #swarm_ee[im]=np.sqrt(0.5*(swarm_exx[im]**2+swarm_eyy[im]**2+2*swarm_exy[im]**2) ) 
            #update its position
            swarm_x[im]+=swarm_u[im]*dt
            swarm_y[im]+=swarm_v[im]*dt
            if swarm_x[im]>=Lx: swarm_x[im]-=Lx #periodic b.c. on right side
            if swarm_x[im]<0:   swarm_x[im]+=Lx #periodic b.c. on left side
            #assign effective visc
            #swarm_eta[im],swarm_sr_dis[im],swarm_sr_diff[im],swarm_sr_gbs[im],swarm_sr_lowT[im]\
            #=viscosity(swarm_ee[im],background_T,swarm_mat[im],swarm_gs[im])
            #assign pressure
            #NNNP=NNP(swarm_r[im],swarm_s[im])
            #swarm_p_dyn[im]=NNNP.dot(p[iconP[0:mP,iel]])
            if swarm_mat[im]<3: swarm_gs[im]=gs_evolution(swarm_gs[im],swarm_ee[im],egs,background_T,dt)
        #end for


        swarmgsfile.write("%f %e %e %e \n" %(istep+iter/200,min(swarm_gs),max(swarm_gs),np.mean(swarm_gs)))
        swarmgsfile.flush()

        swarm_total_strainxx[:]+=swarm_exx[:]*dt
        swarm_total_strainyy[:]+=swarm_eyy[:]*dt
        swarm_total_strainxy[:]+=swarm_exy[:]*dt
        swarm_total_strain_eff[:]=np.sqrt(0.5*(swarm_total_strainxx[:]**2+\
                                               swarm_total_strainyy[:]**2)+\
                                               swarm_total_strainxy[:]**2)
    
        #assign dev stress values
        #swarm_tauxx[im]=2*exxm*swarm_eta[im]
        #swarm_tauyy[im]=2*eyym*swarm_eta[im]
        #swarm_tauxy[im]=2*exym*swarm_eta[im]
        #swarm_sigmaxx[:]=-swarm_p_dyn[:]+swarm_tauxx[:]
        #swarm_sigmayy[:]=-swarm_p_dyn[:]+swarm_tauyy[:]
        #swarm_sigmaxy[:]=               +swarm_tauxy[:]
        #swarm_sigma_angle[:]=0.5*np.arctan(2*swarm_sigmaxy[:]/(swarm_sigmayy[:]-swarm_sigmaxx[:])) 
        #swarm_tau_eff[:]=np.sqrt(0.5*(swarm_tauxx[:]**2+swarm_tauyy[:]**2+2*swarm_tauxy[:]**2))
        #swarm_tau_angle[:]=0.5*np.arctan(2*swarm_tauxy[:]/(swarm_tauyy[:]-swarm_tauxx[:])) 
        #swarm_sigma1[:]=(swarm_sigmaxx[:]+swarm_sigmayy[:])/2. \
        #               + np.sqrt( (swarm_sigmaxx[:]-swarm_sigmayy[:])**2/4 +swarm_sigmaxy[:]**2 ) 
        #swarm_sigma2[:]=(swarm_sigmaxx[:]+swarm_sigmayy[:])/2. \
        #               - np.sqrt( (swarm_sigmaxx[:]-swarm_sigmayy[:])**2/4 +swarm_sigmaxy[:]**2 ) 
    
        logfile.write("     advect markers: %.3f s\n" % (time.time() - start))

        time_advect_markers+=time.time() - start
    
        #####################################################################
        # interpolate pressure onto velocity grid points
        #####################################################################
        start = time.time()
    
        q=np.zeros(NV,dtype=np.float64)
        counter=np.zeros(NV,dtype=np.float64)
    
        for iel in range(0,nel):
            q[iconV[0,iel]]=p[iconP[0,iel]]
            q[iconV[1,iel]]=p[iconP[1,iel]]
            q[iconV[2,iel]]=p[iconP[2,iel]]
            q[iconV[3,iel]]=p[iconP[3,iel]]
            q[iconV[4,iel]]=(p[iconP[0,iel]]+p[iconP[1,iel]])*0.5
            q[iconV[5,iel]]=(p[iconP[1,iel]]+p[iconP[2,iel]])*0.5
            q[iconV[6,iel]]=(p[iconP[2,iel]]+p[iconP[3,iel]])*0.5
            q[iconV[7,iel]]=(p[iconP[3,iel]]+p[iconP[0,iel]])*0.5
            q[iconV[8,iel]]=(p[iconP[0,iel]]+p[iconP[1,iel]]+p[iconP[2,iel]]+p[iconP[3,iel]])*0.25
    
        logfile.write("     project p on Q2: %.3f s\n" % (time.time() - start))
    
        #####################################################################
        # pass swarm arrays to function for analysis/stats, write to file
        #####################################################################
        start = time.time()

        write_history(histfile,background_sr,tempdegC,total_time,istep,\
                      swarm_mat,swarm_total_strain_eff,swarm_eta,\
                      swarm_tau_eff,swarm_ee,swarm_gs)

        logfile.write("     write_history: %.3f s \n" % (time.time() - start))

        #####################################################################
        # plot of solution
        #####################################################################
    
        start = time.time()
    
        #if istep==0 or istep==nstep-1 or total_time>tfinal:
        if True:


            np.savetxt(output_folder+'swarm_gs_tau_{:04d}.ascii'.format(istep),\
                       np.array([swarm_gs,swarm_tauNR]).T)
            np.savetxt(output_folder+'swarm_gs_tau_mid{:04d}.ascii'.format(istep),\
                       np.array([swarm_gs[swarm_mat==2],swarm_tauNR[swarm_mat==2]]).T)
            np.savetxt(output_folder+'swarm_x_y_eta_{:04d}.ascii'.format(istep),\
                       np.array([swarm_x,swarm_y,swarm_eta]).T)
            np.savetxt(output_folder+'mesh_x_y_eta_{:04d}.ascii'.format(istep),\
                       np.array([xc,yc,eta_elemental]).T)

        
            filename = output_folder+'solution_{:04d}.vtu'.format(istep)
            vtufile=open(filename,"w")
            vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
            vtufile.write("<UnstructuredGrid> \n")
            vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(NV,nel))
            #####
            vtufile.write("<Points> \n")
            vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'> \n")
            for i in range(0,NV):
                vtufile.write("%10e %10e %10e \n" %(xV[i],yV[i],0.))
            vtufile.write("</DataArray>\n")
            vtufile.write("</Points> \n")
            #####
            vtufile.write("<CellData Scalars='scalars'>\n")
            #--
            vtufile.write("<DataArray type='Float32' Name='viscosity' Format='ascii'> \n")
            for iel in range (0,nel):
                vtufile.write("%10e\n" % eta_elemental[iel]) 
            vtufile.write("</DataArray>\n")
            #--
            vtufile.write("</CellData>\n")
            #####
            vtufile.write("<PointData Scalars='scalars'>\n")
            #--
            vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Name='velocity (cm/year)' Format='ascii'> \n")
            for i in range(0,NV):
                vtufile.write("%10e %10e %10e \n" %(u[i]/cm*year,v[i]/cm*year,0.))
            vtufile.write("</DataArray>\n")
            #--
            vtufile.write("<DataArray type='Float32' Name='pressure' Format='ascii'> \n")
            for i in range(0,NV):
                vtufile.write("%10e \n" %q[i])
            vtufile.write("</DataArray>\n")
            #--
            vtufile.write("<DataArray type='Float32' Name='exx' Format='ascii'> \n")
            for i in range (0,NV):
                vtufile.write("%10e\n" % exx[i])
            vtufile.write("</DataArray>\n")
            #--
            vtufile.write("<DataArray type='Float32' Name='eyy' Format='ascii'> \n")
            for i in range (0,NV):
                vtufile.write("%10e\n" % eyy[i])
            vtufile.write("</DataArray>\n")
            #--
            vtufile.write("<DataArray type='Float32' Name='exy' Format='ascii'> \n")
            for i in range (0,NV):
                vtufile.write("%10e\n" % exy[i])
            vtufile.write("</DataArray>\n")
            #--
            vtufile.write("</PointData>\n")
            #####
            vtufile.write("<Cells>\n")
            #--
            vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
            for iel in range (0,nel):
                vtufile.write("%d %d %d %d %d %d %d %d %d\n" %(iconV[0,iel],iconV[1,iel],iconV[2,iel],\
                                                               iconV[3,iel],iconV[4,iel],iconV[5,iel],\
                                                               iconV[6,iel],iconV[7,iel],iconV[8,iel]))
            vtufile.write("</DataArray>\n")
            #--
            vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
            for iel in range (0,nel):
                vtufile.write("%d \n" %((iel+1)*9))
            vtufile.write("</DataArray>\n")
            #--
            vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
            for iel in range (0,nel):
                vtufile.write("%d \n" %28)
            vtufile.write("</DataArray>\n")
            #--
            vtufile.write("</Cells>\n")
            #####
            vtufile.write("</Piece>\n")
            vtufile.write("</UnstructuredGrid>\n")
            vtufile.write("</VTKFile>\n")
            vtufile.close()
       
            #---------------------------------------------------------
 
            filename = output_folder+'swarm_{:04d}.vtu'.format(istep)
            vtufile=open(filename,"w")
            vtufile.write("<VTKFile type='UnstructuredGrid' version='0.1' byte_order='BigEndian'> \n")
            vtufile.write("<UnstructuredGrid> \n")
            vtufile.write("<Piece NumberOfPoints=' %5d ' NumberOfCells=' %5d '> \n" %(nmarker,nmarker))
            vtufile.write("<PointData Scalars='scalars'>\n")
            #--
            vtufile.write("<DataArray type='Float32' Name='mat' Format='ascii'>\n")
            for i in range(0,nmarker):
                vtufile.write("%3e \n" %swarm_mat[i])
            vtufile.write("</DataArray>\n")
            #--
            #vtufile.write("<DataArray type='Float32' Name='iel' Format='ascii'>\n")
            #for i in range(0,nmarker):
            #    vtufile.write("%3e \n" %swarm_iel[i])
            #vtufile.write("</DataArray>\n")
            #--
            vtufile.write("<DataArray type='Float32' Name='paint' Format='ascii'>\n")
            for i in range(0,nmarker):
                vtufile.write("%3e \n" %swarm_paint[i])
            vtufile.write("</DataArray>\n")
            #--
            vtufile.write("<DataArray type='Float32' Name='total strain (xx)' Format='ascii'>\n")
            for i in range(0,nmarker):
                vtufile.write("%3e \n" %swarm_total_strainxx[i])
            vtufile.write("</DataArray>\n")
            #--
            vtufile.write("<DataArray type='Float32' Name='total strain (yy)' Format='ascii'>\n")
            for i in range(0,nmarker):
                vtufile.write("%3e \n" %swarm_total_strainyy[i])
            vtufile.write("</DataArray>\n")
            #--
            vtufile.write("<DataArray type='Float32' Name='total strain (xy)' Format='ascii'>\n")
            for i in range(0,nmarker):
                vtufile.write("%3e \n" %swarm_total_strainxy[i])
            vtufile.write("</DataArray>\n")
            #--
            vtufile.write("<DataArray type='Float32' Name='total strain (eff.)' Format='ascii'>\n")
            for i in range(0,nmarker):
                vtufile.write("%3e \n" % swarm_total_strain_eff[i] )
            vtufile.write("</DataArray>\n")
            #--
            vtufile.write("<DataArray type='Float32' Name='grain size (microns)' Format='ascii'>\n")
            for i in range(0,nmarker):
                vtufile.write("%3e \n" % swarm_gs[i] )
            vtufile.write("</DataArray>\n")
            #--
            vtufile.write("<DataArray type='Float32' Name='sr (dis)' Format='ascii'>\n")
            for i in range(0,nmarker):
                vtufile.write("%3e \n" % swarm_sr_dis[i] )
            vtufile.write("</DataArray>\n")
            #--
            vtufile.write("<DataArray type='Float32' Name='sr (diff)' Format='ascii'>\n")
            for i in range(0,nmarker):
                vtufile.write("%3e \n" % swarm_sr_diff[i] )
            vtufile.write("</DataArray>\n")
            #--
            vtufile.write("<DataArray type='Float32' Name='sr (gbs)' Format='ascii'>\n")
            for i in range(0,nmarker):
                vtufile.write("%3e \n" % swarm_sr_gbs[i] )
            vtufile.write("</DataArray>\n")
            #--
            vtufile.write("<DataArray type='Float32' Name='sr (lowT)' Format='ascii'>\n")
            for i in range(0,nmarker):
                vtufile.write("%3e \n" % swarm_sr_lowT[i] )
            vtufile.write("</DataArray>\n")
            #--
            vtufile.write("<DataArray type='Float32' Name='strain rate (xx)' Format='ascii'>\n")
            for i in range(0,nmarker):
                vtufile.write("%3e \n" %swarm_exx[i])
            vtufile.write("</DataArray>\n")
            #--
            vtufile.write("<DataArray type='Float32' Name='strain rate (yy)' Format='ascii'>\n")
            for i in range(0,nmarker):
                vtufile.write("%3e \n" %swarm_eyy[i])
            vtufile.write("</DataArray>\n")
            #--
            vtufile.write("<DataArray type='Float32' Name='strain rate (xy)' Format='ascii'>\n")
            for i in range(0,nmarker):
                vtufile.write("%3e \n" %swarm_exy[i])
            vtufile.write("</DataArray>\n")
            #--
            vtufile.write("<DataArray type='Float32' Name='strain rate (eff.)' Format='ascii'>\n")
            for i in range(0,nmarker):
                vtufile.write("%3e \n" % swarm_ee[i] )
            vtufile.write("</DataArray>\n")
            #--
            vtufile.write("<DataArray type='Float32' Name='T from shear heating' Format='ascii'>\n")
            for i in range(0,nmarker):
                vtufile.write("%3e \n" % (2*swarm_eta[im]*swarm_ee[i]**2/3000/1250*total_time) )
            vtufile.write("</DataArray>\n")

            #--
            #vtufile.write("<DataArray type='Float32' Name='tauxx (MPa)' Format='ascii'>\n")
            #for i in range(0,nmarker):
            #    vtufile.write("%3e \n" %(swarm_tauxx[i]/MPa))
            #vtufile.write("</DataArray>\n")
            #--
            #vtufile.write("<DataArray type='Float32' Name='tauyy (MPa)' Format='ascii'>\n")
            #for i in range(0,nmarker):
            #    vtufile.write("%3e \n" %(swarm_tauyy[i]/MPa))
            #vtufile.write("</DataArray>\n")
            #--
            #vtufile.write("<DataArray type='Float32' Name='tauxy (MPa)' Format='ascii'>\n")
            #for i in range(0,nmarker):
            #    vtufile.write("%3e \n" %(swarm_tauxy[i]/MPa))
            #vtufile.write("</DataArray>\n")
            #--
            #vtufile.write("<DataArray type='Float32' Name='tau (eff.) (MPa)' Format='ascii'>\n")
            #for i in range(0,nmarker):
            #    vtufile.write("%3e \n" % (swarm_tau_eff[i]/MPa))
            #vtufile.write("</DataArray>\n")
            #--
            #vtufile.write("<DataArray type='Float32' Name='tau angle' Format='ascii'>\n")
            #for i in range(0,nmarker):
            #    vtufile.write("%3e \n" % (swarm_tau_angle[i]/np.pi*180))
            #vtufile.write("</DataArray>\n")
            #--
            #vtufile.write("<DataArray type='Float32' Name='sigma angle (abs)' Format='ascii'>\n")
            #for i in range(0,nmarker):
            #    vtufile.write("%3e \n" % (np.abs(swarm_sigma_angle[i])/np.pi*180))
            #vtufile.write("</DataArray>\n")
            #--
            #vtufile.write("<DataArray type='Float32' Name='sigma 1 (MPa)' Format='ascii'>\n")
            #for i in range(0,nmarker):
            #    vtufile.write("%e \n" % (swarm_sigma1[i]/MPa))
            #vtufile.write("</DataArray>\n")
            #--
            #vtufile.write("<DataArray type='Float32' Name='sigma 2 (MPa)' Format='ascii'>\n")
            #for i in range(0,nmarker):
            #    vtufile.write("%e \n" % (swarm_sigma2[i]/MPa))
            #vtufile.write("</DataArray>\n")
            #--
            #vtufile.write("<DataArray type='Float32' Name='sigma_1 (dir)' NumberOfComponents='3' Format='ascii'> \n")
            #for i in range(0,nmarker):
            #    vtufile.write("%10e %10e %10e \n" %( np.cos(swarm_sigma_angle[i]),np.sin(swarm_sigma_angle[i]),0) )
            #vtufile.write("</DataArray>\n")
            #--
            #vtufile.write("<DataArray type='Float32' Name='sigma_2 (dir)' NumberOfComponents='3' Format='ascii'> \n")
            #for i in range(0,nmarker):
            #    vtufile.write("%10e %10e %10e \n" %( np.cos(swarm_sigma_angle[i]+np.pi/2),np.sin(swarm_sigma_angle[i]+np.pi/2),0) )
            #vtufile.write("</DataArray>\n")
            #--
            vtufile.write("<DataArray type='Float32' Name='viscosity (Pa.s)' Format='ascii'>\n")
            for i in range(0,nmarker):
                vtufile.write("%10e \n" %swarm_eta[i])
            vtufile.write("</DataArray>\n")
            #--
            #vtufile.write("<DataArray type='Float32' Name='pressure dyn. (MPa)' Format='ascii'>\n")
            #for i in range(0,nmarker):
            #    vtufile.write("%10e \n" %(swarm_p_dyn[i]/MPa))
            #vtufile.write("</DataArray>\n")
            #--
            vtufile.write("<DataArray type='Float32' Name='r,s,t' NumberOfComponents='3' Format='ascii'>\n")
            for i in range(0,nmarker):
                vtufile.write("%5e %5e %5e \n" %(swarm_r[i],swarm_s[i],0.))
            vtufile.write("</DataArray>\n")
            #--
            vtufile.write("<DataArray type='Float32' Name='velocity (cm/year)' NumberOfComponents='3' Format='ascii'>\n")
            for i in range(0,nmarker):
                vtufile.write("%5e %5e %5e \n" %(swarm_u[i]/cm*year,swarm_v[i]/cm*year,0.))
            vtufile.write("</DataArray>\n")
            #--
            vtufile.write("</PointData>\n")
            vtufile.write("<Points> \n")
            vtufile.write("<DataArray type='Float32' NumberOfComponents='3' Format='ascii'>\n")
            for i in range(0,nmarker):
                vtufile.write("%10e %10e %10e \n" %(swarm_x[i],swarm_y[i],0.))
            vtufile.write("</DataArray>\n")
            vtufile.write("</Points> \n")
            vtufile.write("<Cells>\n")
            vtufile.write("<DataArray type='Int32' Name='connectivity' Format='ascii'> \n")
            for i in range(0,nmarker):
                vtufile.write("%d " % i)
            vtufile.write("</DataArray>\n")
            vtufile.write("<DataArray type='Int32' Name='offsets' Format='ascii'> \n")
            for i in range(0,nmarker):
                vtufile.write("%d " % (i+1))
            vtufile.write("</DataArray>\n")
            vtufile.write("<DataArray type='Int32' Name='types' Format='ascii'>\n")
            for i in range(0,nmarker):
                vtufile.write("%d " % 1)
            vtufile.write("</DataArray>\n")
            vtufile.write("</Cells>\n")
            vtufile.write("</Piece>\n")
            vtufile.write("</UnstructuredGrid>\n")
            vtufile.write("</VTKFile>\n")
            vtufile.close()
        
            logfile.write("     export to vtu: %.3f s \n" % (time.time() - start))
        
            #----------------------------------
            if False: 
               start = time.time()
               plt.figure(figsize=(20,5))
               plt.scatter(swarm_x,swarm_y,c=np.log10(swarm_eta),s=1)
               plt.title('effective viscosity')
               plt.xlim([0,Lx])
               plt.ylim([0,Ly])
               filename = 'swarm_eta_{:04d}.png'.format(istep)
               plt.savefig(filename,bbox_inches='tight')
               plt.figure(figsize=(20,5))
               plt.scatter(swarm_x,swarm_y,c=swarm_mat,s=1)
               plt.title('material')
               plt.xlim([0,Lx])
               plt.ylim([0,Ly])
               filename = 'swarm_mat_{:04d}.png'.format(istep)
               plt.savefig(filename,bbox_inches='tight')
               logfile.write("     export to png: %.3f s \n" % (time.time() - start))
            #end if

        #end if

        if total_time>tfinal:
           logfile.write("*****tfinal reached*****\n")
           break
    
    ###################################################################################################
    # end for istep
    ###################################################################################################

    print('  |time_localise_markers= ',time_localise_markers)
    print('  |time_advect_markers=   ',time_advect_markers)
    print('  |time_build_matrix=     ',time_build_matrix)
    print('  |time_solve_system=     ',time_solve_system)
    print('  |time_compute_viscosity=',time_compute_viscosity)
    print('  |time_compute_sr=       ',time_compute_sr)
    
    logfile.write("-----------------------------\n")
    logfile.write(f"-------end ------\n")
    logfile.write("-----------------------------\n")

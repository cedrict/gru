import numpy as np

###################################################################################################

def write_history(histfilename,sr,temp,total_time,istep,\
                  swarm_mat,\
                  swarm_total_strain_eff,\
                  swarm_eta,\
                  swarm_tau_eff,\
                  swarm_ee,\
                  swarm_grainsize):
    
        if istep==0:
           histfile=open(histfilename,"w")
           histfile.write("#1-istep 2-total_time 3-strain12 4-vis12 5-sr12 6-grainsize12 7-strain2 8-vis2 9-sr2 10-grainsize2  11-bg_sr 12-temp \n")
               
           # histfile.write("# col  2: total_time")
           # histfile.write("# col  3: strain12")
           # histfile.write("# col  4: vis12")
           # histfile.write("# col  5: sr12")
           # histfile.write("# col  6: grainsize12")
           # histfile.write("# col  7: strain2")
           # histfile.write("# col  8: vis2")
           # histfile.write("# col  9: sr2")
           # histfile.write("# col  10: grainsize2")
           # histfile.write("# col  11: back ground strain rate")
           # histfile.write("# col  12: back ground temperature")
           
        else:
           histfile=open(histfilename,"a")
           
        #------------------------------------------------------
        # compute mean strain, viscosity and strain rate in shear zone and a small portion of the shear zone
    
        count_mat1 = 0  
        count_mat2 = 0
     
        for i in range (len(swarm_mat)):
            if swarm_mat[i] == 1:    # define the mean for the entire mat 1
                count_mat1+=1
                mat1 = np.where(swarm_mat == 1 )
            if swarm_mat[i] ==  2:    # define the mean for the mat 2
                count_mat2+=1
                mat2 = np.where(swarm_mat == 2)
             
        totstrain12= (count_mat1*np.mean(swarm_total_strain_eff[mat1])+count_mat2*np.mean(swarm_total_strain_eff[mat2]))/(count_mat1+count_mat2)
        visc12 = (count_mat1*np.mean(swarm_eta[mat1])+count_mat2*np.mean(swarm_eta[mat2]))/(count_mat1+count_mat2)
        sr12 = (count_mat1*np.mean(swarm_ee[mat1])+count_mat2*np.mean(swarm_ee[mat2]))/(count_mat1+count_mat2)
        gs12 = (count_mat1*np.mean(swarm_grainsize[mat1])+count_mat2*np.mean(swarm_grainsize[mat2]))/(count_mat1+count_mat2)
        totstrain2= np.mean(swarm_total_strain_eff[mat2])
        visc2 = np.mean(swarm_eta[mat2])
        sr2 = np.mean(swarm_ee[mat2])
        gs2 = np.mean(swarm_grainsize[mat2])
    
        #------------------------------------------------------
    
        histfile.write("%d %e %e %e %e %e %e %e %e %e %e %e \n" %(\
                        istep,      \
                        total_time, \
                        totstrain12,  \
                        visc12,  \
                        sr12,  \
                        gs12,   \
                        totstrain2,  \
                        visc2, \
                        sr2,  \
                        gs2, \
                        sr,  \
                        temp))
    
        histfile.close()

###################################################################################################

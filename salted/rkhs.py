import sys
import numpy as np
import time
import h5py

from salted.sys_utils import read_system, get_atom_idx, get_conf_range

def build():

    sys.path.insert(0, './')
    import inp
    
    if inp.parallel:
        from mpi4py import MPI
        # MPI information
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    #    print('This is task',rank+1,'of',size)
    else:
        rank = 0
    
    saltedname = inp.saltedname
    
    if inp.field==True:
        kdir = inp.saltedpath+"kernels_"+saltedname+"_field"
    else:
        kdir = inp.saltedpath+"kernels_"+saltedname
    kdir += '/'
    
    species, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()
    
    atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)
    
    # number of sparse environments
    M = inp.Menv
    zeta = inp.z
    eigcut = inp.eigcut
    if rank == 0:
        print("M =", M, "eigcut =", eigcut)
        print("zeta =", zeta)
        print("Computing RKHS of symmetry-adapted sparse kernel approximations...")
    sdir = inp.saltedpath+'equirepr_'+inp.saltedname+'/'
    
    # Distribute structures to tasks
    if inp.parallel:
        conf_range = get_conf_range(rank,size,ndata,list(range(ndata)))
        conf_range = comm.scatter(conf_range,root=0)
        print('Task',rank+1,'handles the following structures:',conf_range,flush=True)
    else:
        conf_range = list(range(ndata))
    
    power_env_sparse = {}
    if inp.field: power_env_sparse2 = {}
    kernel0_nm = {}
    
    power = h5py.File(sdir+"FEAT-0.h5",'r')["descriptor"][conf_range,:]
    nfeat = power.shape[-1]
    if inp.field:
        power2 = h5py.File(sdir+"FEAT-0_field.h5",'r')["descriptor"][conf_range,:]
        nfeat2 = power2.shape[-1]
    Mspe = {}
    
    for spe in species:
        if rank == 0: print("lambda = 0", "species:", spe)
        start = time.time()
    
        # compute sparse kernel K_MM for each atomic species 
        power_env_sparse[spe] = h5py.File(sdir+"FEAT-0-M-"+str(M)+".h5",'r')[spe][:]
        if inp.field: power_env_sparse2[spe] = h5py.File(sdir+"FEAT-0-M-"+str(M)+"_field.h5",'r')[spe][:]
        Mspe[spe] = power_env_sparse[spe].shape[0]
    
        V = np.load(kdir+"spe"+str(spe)+"_l"+str(0)+"/M"+str(M)+"_zeta"+str(zeta)+"/projector.npy")
        
        # compute feature vector Phi associated with the RKHS of K_NM * K_MM^-1 * K_NM^T
        for i,iconf in enumerate(conf_range):
            kernel0_nm[(iconf,spe)] = np.dot(power[i,atom_idx[(iconf,spe)]],power_env_sparse[spe].T)
            if inp.field:
                kernel_nm = np.dot(power2[i,atom_idx[(iconf,spe)]],power_env_sparse2[spe].T)
                kernel_nm += kernel0_nm[(iconf,spe)]
                kernel_nm *= kernel0_nm[(iconf,spe)]**(zeta-1)
            else:
                kernel_nm = kernel0_nm[(iconf,spe)]**zeta
            psi_nm = np.real(np.dot(kernel_nm,V))
            np.save(kdir+"spe"+str(spe)+"_l"+str(0)+"/M"+str(M)+"_zeta"+str(zeta)+"/psi-nm_conf"+str(iconf)+".npy",psi_nm)
        if rank == 0: print((time.time()-start)/60.0)
    
    # lambda>0
    for l in range(1,llmax+1):
    
        # load power spectrum
        if rank == 0: print("loading lambda =", l)
        power = h5py.File(sdir+"FEAT-"+str(l)+".h5",'r')["descriptor"][conf_range,:]
        nfeat = power.shape[-1]
        if inp.field:
            power2 = h5py.File(sdir+"FEAT-"+str(l)+"_field.h5",'r')["descriptor"][conf_range,:]
            nfeat2 = power2.shape[-1]
    
    #    if inp.field:
    #        power = h5py.File(sdir+"FEAT-"+str(l)+"_field.h5",'r')["descriptor"][conf_range,:]
    #        power2 = h5py.File(sdir+"FEAT-"+str(l)+".h5",'r')["descriptor"][conf_range,:]
    #        nfeat = power.shape[-1]
    #    else:
    #        power = h5py.File(sdir+"FEAT-"+str(l)+".h5",'r')["descriptor"][conf_range,:]
    #        nfeat = power.shape[-1]
    
        for spe in species:
            if rank == 0: print("lambda = ", l, "species:", spe)
            start = time.time()
    
            # get sparse feature vector for each atomic species
            power_env_sparse[spe] = h5py.File(sdir+"FEAT-"+str(l)+"-M-"+str(M)+".h5",'r')[spe][:]
            if inp.field: power_env_sparse2[spe] = h5py.File(sdir+"FEAT-"+str(l)+"-M-"+str(M)+"_field.h5",'r')[spe][:]
            V = np.load(kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_zeta"+str(zeta)+"/projector.npy") 
    
            # compute feature vector Phi associated with the RKHS of K_NM * K_MM^-1 * K_NM^T
            for i,iconf in enumerate(conf_range):
                kernel_nm = np.dot(power[i,atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*(2*l+1),nfeat),power_env_sparse[spe].T) 
                if inp.field: kernel_nm += np.dot(power2[i,atom_idx[(iconf,spe)]].reshape(natom_dict[(iconf,spe)]*(2*l+1),nfeat2),power_env_sparse2[spe].T) 
                for i1 in range(natom_dict[(iconf,spe)]):
                    for i2 in range(Mspe[spe]):
                        kernel_nm[i1*(2*l+1):i1*(2*l+1)+2*l+1][:,i2*(2*l+1):i2*(2*l+1)+2*l+1] *= kernel0_nm[(iconf,spe)][i1,i2]**(zeta-1)
                psi_nm = np.real(np.dot(kernel_nm,V))
                np.save(kdir+"spe"+str(spe)+"_l"+str(l)+"/M"+str(M)+"_zeta"+str(zeta)+"/psi-nm_conf"+str(iconf)+".npy",psi_nm)
            if rank == 0: print((time.time()-start)/60.0)

    return

if __name__ == "__main__":
    build()
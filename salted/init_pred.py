import os
import sys
import time
import os.path as osp

import h5py
import numpy as np
from scipy import special

from salted import basis

def build():

    sys.path.insert(0, './')
    import inp

    saltedname = inp.saltedname
    species = inp.species 
    ncut = inp.ncut 
    M = inp.Menv
    zeta = inp.z
    reg = inp.regul
    sparsify = inp.sparsify
  
    # read basis
    [lmax,nmax] = basis.basiset(inp.dfbasis)
    llist = []
    nlist = []
    for spe in species:
        llist.append(lmax[spe])
        for l in range(lmax[spe]+1):
            nlist.append(nmax[(spe,l)])
    lmax_max = max(llist)

    charge_integrals = {}
    if inp.qmcode=="cp2k":

        # get basis set info from CP2K BASIS_LRIGPW_AUXMOLOPT
        alphas = {}
        sigmas = {}
        for spe in species:
            for l in range(lmax[spe]+1):
                avals = np.loadtxt(osp.join(
                    inp.saltedpath, "basis", f"{spe}-{inp.dfbasis}-alphas-L{l}.dat"
                ))
                if nmax[(spe,l)]==1:
                    alphas[(spe,l,0)] = float(avals)
                    sigmas[(spe,l,0)] = np.sqrt(0.5/alphas[(spe,l,0)]) # bohr
                else:
                    for n in range(nmax[(spe,l)]):
                        alphas[(spe,l,n)] = avals[n]
                        sigmas[(spe,l,n)] = np.sqrt(0.5/alphas[(spe,l,n)]) # bohr

        # compute integrals of basis functions (needed to a posteriori correction of the charge)
        for spe in species:
            for l in range(lmax[spe]+1):
                charge_integrals_temp = np.zeros(nmax[(spe,l)])
                for n in range(nmax[(spe,l)]):
                    inner = 0.5*special.gamma(l+1.5)*(sigmas[(spe,l,n)]**2)**(l+1.5)
                    charge_radint = 0.5 * special.gamma(float(l+3)/2.0) / ( (alphas[(spe,l,n)])**(float(l+3)/2.0) )
                    charge_integrals[(spe,l,n)] = charge_radint * np.sqrt(4.0*np.pi) / np.sqrt(inner)
    
    loadstart = time.time()
   
    # Load feature space sparsification information if required 
    if sparsify:
        vfps = {}
        for lam in range(lmax_max+1):
            vfps[lam] = np.load(osp.join(
                inp.saltedpath, f"equirepr_{saltedname}", f"fps{ncut}-{lam}.npy"
            ))

    # Load training feature vectors and RKHS projection matrix
    Vmat,Mspe,power_env_sparse = get_feats_projs(species,lmax)
 
    # load regression weights
    ntrain = int(inp.Ntrain*inp.trainfrac)
    if inp.field:
        weights = np.load(osp.join(
            inp.saltedpath, f"regrdir_{saltedname}_field", f"M{M}_zeta{zeta}", f"weights_N{ntrain}_reg{int(np.log10(reg))}.npy"
        ))
    else:
        weights = np.load(osp.join(
            inp.saltedpath, f"regrdir_{saltedname}", f"M{M}_zeta{zeta}", f"weights_N{ntrain}_reg{int(np.log10(reg))}.npy"
        ))
    
    print("load time:", (time.time()-loadstart))
    
    return [lmax,nmax,lmax_max,weights,power_env_sparse,Mspe,Vmat,vfps,charge_integrals]

if __name__ == "__main__":
    build()

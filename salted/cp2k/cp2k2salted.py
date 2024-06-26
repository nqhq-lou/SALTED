import os
import sys
import math
import numpy as np
from ase.io import read
from scipy import special
from itertools import islice
import copy
import time

from salted import basis
from salted.sys_utils import ParseConfig


inp = ParseConfig().parse_input()

xyzfile = read(inp.system.filename,":")
ndata = len(xyzfile)
species = inp.system.species
[lmax,nmax] = basis.basiset(inp.qm.dfbasis)

dirpath = os.path.join(inp.salted.saltedpath, "coefficients")
# dirpath = os.path.join(inp.salted.saltedpath, "coefficients-efield")
if not os.path.exists(dirpath):
    os.mkdir(dirpath)

# init geometry
for iconf in range(ndata):
    geom = xyzfile[iconf]
    symbols = geom.get_chemical_symbols()
    natoms = len(symbols)
    # compute basis set size
    nRI = 0
    for iat in range(natoms):
        spe = symbols[iat]
        if spe in species:
            for l in range(lmax[spe]+1):
                for n in range(nmax[(spe,l)]):
                    nRI += 2*l+1

    # load density coefficients and check dimension
    coefficients = np.loadtxt(os.path.join(inp.qm.path2qm, f"conf_{iconf+1}", inp.qm.coeffile))
    # coefficients = np.loadtxt(os.path.join(inp.qm.path2qm, f"conf_{iconf+1}", "efield", inp.qm.coeffile))
    if len(coefficients)!=nRI:
        print("ERROR: basis set size does not correspond to size of coefficients vector!")
        sys.exit(0)
    else:
        print("conf", iconf+1, "size =", nRI, flush=True)

    # save coefficients vector in SALTED format
    if natoms%2 != 0:
        coefficients = np.sum(coefficients,axis=1)
    np.save(os.path.join(inp.salted.saltedpath, "coefficients", f"coefficients_conf{iconf}.npy"), coefficients)
    # np.save(os.path.join(inp.salted.saltedpath, "coefficients-efield", f"coefficients_conf{iconf}.npy"), coefficients)

    # save overlap matrix in SALTED format
    overlap = np.zeros((nRI, nRI)).astype(np.double)
    for i in range(nRI):
       offset = 4 + i*((nRI+1)*8)
       overlap[:, i] = np.fromfile(os.path.join(
           inp.qm.path2qm, f"conf_{iconf+1}", inp.qm.ovlpfile
        ), dtype=np.float64, offset = offset, count=nRI)

    dirpath = os.path.join(inp.salted.saltedpath, "overlaps")
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    np.save(os.path.join(inp.salted.saltedpath, "overlaps", f"overlap_conf{iconf}.npy"), overlap)

    ## save projections vector in SALTED format
    #projections = np.dot(overlap,coefficients)
    #dirpath = os.path.join(inp.salted.saltedpath, "projections")
    #if not os.path.exists(dirpath):
    #    os.mkdir(dirpath)
    #np.save(inp.salted.saltedpath+"projections/projections_conf"+str(iconf)+".npy",projections)

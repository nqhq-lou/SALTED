import sys
import time

import numpy as np
from pyscf import gto
from ase.io import read
from scipy import special

import basis  # WARNING: relative import
sys.path.insert(0, './')
import inp

# read species
spelist = inp.species
spe_dict = {}
for i in range(len(spelist)):
    spe_dict[i] = spelist[i]

# read basis
[lmax,nmax] = basis.basiset(inp.dfbasis)

llist = []
nlist = []
for spe in spelist:
    llist.append(lmax[spe])
    for l in range(lmax[spe]+1):
        nlist.append(nmax[(spe,l)])
llmax = max(llist)
nnmax = max(nlist)

# read system
xyzfile = read(inp.filename,":")
ndata = len(xyzfile)

hart2kcal = 627.5096080305927
bohr2ang = 0.52917721067121

#======================= system parameters
atomic_symbols = []
atomic_valence = []
natoms = np.zeros(ndata,int) 
for i in range(len(xyzfile)):
    atomic_symbols.append(xyzfile[i].get_chemical_symbols())
    atomic_valence.append(xyzfile[i].get_atomic_numbers())
    natoms[i] = int(len(atomic_symbols[i]))
natmax = max(natoms)

def complex_to_real_transformation(sizes):
    """Transformation matrix from complex to real spherical harmonics"""
    matrices = []
    for i in range(len(sizes)):
        lval = (sizes[i]-1)/2
        st = (-1.0)**(lval+1)
        transformation_matrix = np.zeros((sizes[i],sizes[i]),dtype=complex)
        for j in range( int((sizes[i]-1)/2) ):
            transformation_matrix[j][j] = 1.0j
            transformation_matrix[j][sizes[i]-j-1] = st*1.0j
            transformation_matrix[sizes[i]-j-1][j] = 1.0
            transformation_matrix[sizes[i]-j-1][sizes[i]-j-1] = st*-1.0
            st = st * -1.0
        transformation_matrix[int((sizes[i]-1)/2)][int((sizes[i]-1)/2)] = np.sqrt(2.0)
        transformation_matrix /= np.sqrt(2.0)
        matrices.append(transformation_matrix)
    return matrices

def sph_zproj(theta,phi,lam):
    """Project on the spherical harmonics evaluated at d_ij direction"""
    sph_dij = np.zeros(2*lam+1,complex)
    for mu in range(2*lam+1):
        sph_dij[mu] = special.sph_harm(mu-lam,lam,phi,theta)
    CR = complex_to_real_transformation([2*lam+1])[0]
    return np.real(np.dot(CR,sph_dij))

def radint0(alp):
    """Centered Coulomb integral from 0 to infinity"""
    sigma = np.sqrt(1.0/(2*alp))
    inner = 0.5*special.gamma(1.5)*(sigma**2)**(1.5)
    rint0 = 0.5/alp
    return rint0/np.sqrt(inner)

def radint1(lval,alp,d):
    """Coulomb-Legendre integral from 0 to dij"""
    sigma = np.sqrt(1.0/(2*alp))
    inner = 0.5*special.gamma(lval+1.5)*(sigma**2)**(lval+1.5)
    rint1 = 0.5*alp**(-1.5-lval)*(special.gamma(1.5+lval)
            -special.gamma(1.5 + lval)*special.gammaincc(lval+1.5,alp*d**2))
    return rint1/np.sqrt(inner)

def radint2(lval,alp,d):
    """Coulomb-Legendre integral from dij to infinity"""
    sigma = np.sqrt(1.0/(2*alp))
    inner = 0.5*special.gamma(lval+1.5)*(sigma**2)**(lval+1.5)
    rint2 = 0.5*np.exp(-alp*d**2)/alp
    return rint2/np.sqrt(inner)

f = open("hartree_energy.dat","w")
g = open("external_energy.dat","w")
print("Computing Hartree and external energy...")
itest=0
for iconf in range(ndata):
    print("conf. number:", iconf+1, "/", ndata)
    atoms = atomic_symbols[iconf]
    valences = atomic_valence[iconf]
    nele = np.sum(valences)
    natoms = len(atoms)
    # Load projections and overlaps
    rcoeffs = np.load(os.path.join(
        inp.path2qm, inp.coefdir, f"coefficients_conf{iconf}.npy"
    ))
    ref_coeffs = np.zeros((natoms,llmax+1,nnmax,2*llmax+1))
    ref_rho = np.zeros(len(rcoeffs))
    icoeff = 0
    for iat in range(natoms):
        for l in range(lmax[atoms[iat]]+1):
            for n in range(nmax[(atoms[iat],l)]):
                for im in range(2*l+1):
                    ref_coeffs[iat,l,n,im] = rcoeffs[icoeff]
                    if l==1:
                        if im != 2:
                           ref_rho[icoeff+1] = rcoeffs[icoeff]
                        elif im == 2:
                           ref_rho[icoeff-2] = rcoeffs[icoeff]

                    else:
                        ref_rho[icoeff] = rcoeffs[icoeff]
                    icoeff +=1
    geom = xyzfile[iconf]
    coords = geom.get_positions()
    catoms = []
    for i in range(natoms):
        coord = coords[i]
        catoms.append([atoms[i],(coord[0],coord[1],coord[2])])
    coords /= bohr2ang
    # basis
    mol = gto.M(atom=catoms,basis=inp.qmbasis)
    ribasis = inp.qmbasis+" jkfit" 
    auxmol = gto.M(atom=catoms,basis=ribasis)
    pmol = mol + auxmol
    # RI
    eri2c = auxmol.intor('int2c2e_sph')
    matrix_of_coefficients = np.outer(ref_rho,ref_rho)
    E_H_ref = np.einsum('ij,ji',eri2c,matrix_of_coefficients)*0.5
    # Compute electron-nuclear energy
    E_eN_ref = 0.0
    for iat in range(natoms):
        ref_contr = 0.0
        for jat in range(natoms):
            spe = atoms[jat]
            if iat==jat:
                idx=0
                for n in range(nmax[(spe,0)]):
                    alpha=auxmol._basis[spe][idx][1][0]
                    r0 = radint0(alpha)
                    ref_contr += ref_coeffs[jat,0,n,0]*r0
                    idx+=1
            else:
                dist = coords[iat]-coords[jat]
                dij  = np.linalg.norm(dist)
                theta = np.arccos(dist[2]/dij)
                phi = np.arctan2(dist[1],dist[0])
                idx=0
                for l in range(lmax[spe]+1):
                    wigner_real = sph_zproj(theta,phi,l)*np.sqrt(4.0*np.pi)/float(2*l+1)
                    for n in range(nmax[(spe,l)]):
                        alpha=auxmol._basis[spe][idx][1][0]
                        r1 = radint1(l,alpha,dij)/(dij**(l+1))
                        r2 = radint2(l,alpha,dij)*(dij**l)
                        rotated_coeffs = np.dot(wigner_real,ref_coeffs[jat,l,n,:2*l+1])
                        ref_contr += (r1 + r2)*rotated_coeffs
                        idx+=1
        E_eN_ref += ref_contr*valences[iat]
    E_eN_ref *= -np.sqrt(4.0*np.pi)
    print(E_H_ref,file=f)
    print(E_eN_ref,file=g)
    itest+=1

f.close()
g.close()

# compute error on electrostatic energy
hartree_ref = np.loadtxt("hartree_energy.dat")
external_ref = np.loadtxt("external_energy.dat")
electro_ref = hartree_ref + external_ref 
np.savetxt("electrostatic_energy.dat",electro_ref)

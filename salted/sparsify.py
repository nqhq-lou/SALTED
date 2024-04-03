import numpy  # sorry for this, just for testing new functions
import numpy as np
import sys
import h5py
import os
import os.path as osp
from typing import Union, Literal, Optional, Dict

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from salted.sys_utils import read_system, get_atom_idx



def viz_distribution(
    data:numpy.ndarray,
    slct_idxes:numpy.ndarray,
    species_idxes:dict[str, numpy.ndarray],
    slct_idxes_species:dict[str, numpy.ndarray],
    method:Union[Literal["tsne"], Literal["pca"]]="pca",
    title:Optional[str]=None,
):
    if method.lower() == "pca":
        reducer = PCA(n_components=2)
    elif method.lower() == "tsne":
        reducer = TSNE(n_components=2, init="pca", learning_rate="auto")
    else:
        raise ValueError(f"should have method in ['pca', 'tsne'], but {method=}")

    spe_colors = {spe: f"C{i}" for i, spe in enumerate(slct_idxes_species.keys())}

    fig, axes = plt.subplots(
        1, len(slct_idxes_species)+1,
        figsize=(5*(len(slct_idxes_species) + 1), 5)
    )
    axes = axes.flatten()

    """plot kwargs"""
    s = 5
    alpha = 0.3

    """for all data"""
    ax = axes[0]
    data_reduced = reducer.fit_transform(data)
    for spe, spe_idxes in species_idxes.items():
        ax.scatter(data_reduced[spe_idxes, 0], data_reduced[spe_idxes, 1], c=spe_colors[spe], s=s, alpha=alpha, label=spe)
    ax.scatter(data_reduced[slct_idxes, 0], data_reduced[slct_idxes, 1], c="r", s=2*s, label="selected")
    ax.legend()
    ax.grid()
    ax.set_title(f"all species, {len(slct_idxes)} selected")

    """for each species"""
    for i, spe in enumerate(species_idxes.keys(), 1):
        ax = axes[i]
        spe_idxes = species_idxes[spe]
        slct_idxes_spe = slct_idxes_species[spe]
        data_reduced = reducer.fit_transform(data[spe_idxes])
        ax.scatter(data_reduced[:, 0], data_reduced[:, 1], c=spe_colors[spe], s=s, alpha=alpha, label=spe)
        ax.scatter(data_reduced[slct_idxes_spe, 0], data_reduced[slct_idxes_spe, 1], c="r", s=2*s, label="selected")
        ax.legend()
        ax.grid()
        ax.set_title(f"species {spe}, {len(slct_idxes_spe)} selected")

    if title is not None:
        fig.suptitle(title)
    fig.tight_layout()
    fig = plt.gcf()
    plt.close()
    return fig


def do_fps_new(data:numpy.ndarray, select_cnt:int, seed=None) -> numpy.ndarray:
    """Furthest point sampling for data (shape [nsamples, nfeat])
    """

    assert data.ndim == 2, f"should have data.ndim == 2 by now, but {data.ndim=}"

    select_cnt = int(select_cnt)

    if select_cnt >= data.shape[0]:
        print("WARNING: select_cnt >= data.shape[0], return all idxes", file=sys.stderr)
        return numpy.arange(data.shape[0])
    assert 0 < select_cnt < data.shape[0], f"should have 0 < select_cnt < data_size={data.shape[0]}, but {select_cnt=}"

    if isinstance(seed, int):
        numpy.random.seed(seed)
        start_idx = numpy.random.randint(0, data.shape[0])
    else:
        start_idx = 0
    select_idxes = [None for _ in range(select_cnt)]  # start from a random point
    select_idxes[0] = start_idx  # start from a random point

    data_square = numpy.sum(numpy.square(data).real, axis=-1)  # real, shape (nsamples,)
    """Trick: maintain an all-data to select-data minimum distance square array"""
    to_select_dist_square = data_square + data_square[start_idx] \
        - 2 * numpy.real(numpy.dot(data, data[start_idx].conj()))  # real, shape (nsamples, ...,)
        # - 2 * numpy.real(numpy.sum(data * data[start_idx], axis=-1))  # real, shape (nsamples, ...,)

    for i in range(1, select_cnt):  # evaluate the i-th point
        current_idx = select_idxes[i-1]
        current_data = data[current_idx]
        """calculate the distance square from current_data to all data"""
        to_current_dist_square = data_square + data_square[current_idx] \
            - 2 * numpy.real(numpy.dot(data, current_data))
            # - 2 * numpy.real(numpy.sum(data * current_data, axis=-1))
        """update the minimum distance square from all data to selected data"""
        to_select_dist_square = numpy.minimum(to_select_dist_square, to_current_dist_square)
        """select the next point"""
        next_idx = numpy.argmax(to_select_dist_square)
        select_idxes[i] = next_idx

    return numpy.array(select_idxes, dtype=int)

def do_fps_prepare_l(data:numpy.ndarray, select_cnt:int, seed=None):
    """Do furthest point sampling for l-th feature of data. Could be used for complex data.

    data shape: [nsamples, [mag_num,] nfeat]
    mag_num: -l, -l+1, ..., l-1, l. l is data_l, angualr momentum
    """
    data = data.copy()

    assert select_cnt > 0, f"should have select_cnt > 0, but {select_cnt=}"
    if select_cnt >= data.shape[0]:
        print("WARNING: select_cnt >= data.shape[0], return all idxes", file=sys.stderr)
        return numpy.arange(data.shape[0])
    assert 0 < select_cnt < data.shape[0], f"should have 0 < select_cnt < data_size={data.shape[0]}, but {select_cnt=}"
    if data.ndim == 3:
        assert data.shape[1] % 2 == 1, f"should have data.shape[1] % 2 == 1, but {data.shape[1]=}"
        data = data.reshape(data.shape[0], -1)  # flatten the last dimensions
    elif data.ndim == 2:
        pass
    else:
        raise ValueError(f"should have data.ndim == 2 or 3, but {data.ndim=}")

    return do_fps_new(data, select_cnt, seed)


def sparse_kmeans(data:numpy.ndarray, select_cnt:int, seed=None) -> numpy.ndarray:
    """Furthest point sampling for data (shape [nsamples, nfeat]) using KMeans
    """

    if isinstance(seed, int):
        numpy.random.seed(seed)
    kmeans = KMeans(n_clusters=select_cnt, random_state=seed)
    kmeans.fit(data)
    centroids = kmeans.cluster_centers_

    """get the index of each centroid in data"""
    dist = numpy.square(data.reshape(data.shape[0], 1, -1) - centroids.reshape(1, centroids.shape[0], -1)).sum(axis=-1)
    return numpy.argmin(dist, axis=0)



def sparse_random_mindist(data:numpy.ndarray, select_cnt:int, dist_coeff:float, seed=None):
    """Sparsification by random selection and minimum distance

    data: shape (nsamples, nfeat)

    Notice: min_dist will be scaled by data feature dimension because of the variance of each feature
    """
    if seed is not None:
        numpy.random.seed(seed)
        start_idx = numpy.random.randint(0, data.shape[0])
    else:
        start_idx = 0
    select_idxes = [None for _ in range(select_cnt)]  # start from a random point
    select_idxes[0] = start_idx  # start from a random point

    min_dist_square = numpy.sum(data**2) / data.shape[0] * dist_coeff  # scale by feature dimension
    print(f"scaled min dist: {min_dist_square ** 0.5:.3e}")

    data_square = numpy.sum(numpy.square(data).real, axis=-1)  # real, shape (nsamples,)
    """Trick: maintain an all-data to select-data minimum distance square array"""
    to_select_dist_square = data_square + data_square[start_idx] \
        - 2 * numpy.real(numpy.dot(data, data[start_idx].conj()))  # real, shape (nsamples, ...,)
        # - 2 * numpy.real(numpy.sum(data * data[start_idx], axis=-1))  # real, shape (nsamples, ...,)

    for i in range(1, select_cnt):  # evaluate the i-th point
        current_idx = select_idxes[i-1]
        current_data = data[current_idx]
        """calculate the distance square from current_data to all data"""
        to_current_dist_square = data_square + data_square[current_idx] \
            - 2 * numpy.real(numpy.dot(data, current_data))
            # - 2 * numpy.real(numpy.sum(data * current_data, axis=-1))
        """update the minimum distance square from all data to selected data"""
        to_select_dist_square = numpy.minimum(to_select_dist_square, to_current_dist_square)
        """select the next point"""
        avail_idxes = numpy.where(to_select_dist_square >= min_dist_square)[0]
        if len(avail_idxes) == 0:
            print(f"no available points with distance {dist_coeff=}, sampled {i} points", file=sys.stderr)
            break
        select_idxes[i] = numpy.random.choice(avail_idxes)

    """filter None in select_idxes, and calculate the actual minimum and maximum distance"""
    select_idxes = [idx for idx in select_idxes if idx is not None]
    select_data = data[select_idxes]
    select_data_dist = numpy.square(
        select_data.reshape(len(select_data), 1, -1) - select_data.reshape(1, len(select_data), -1)
    ).sum(axis=(-1)) ** 0.5

    # print(select_data_dist)
    dist_max = select_data_dist.max()
    numpy.fill_diagonal(select_data_dist, numpy.inf)
    dist_min = select_data_dist.min()
    print(f"actual min dist: {dist_min:.3e}\nactual max dist: {dist_max:.3e}")

    return numpy.array(select_idxes, dtype=int)


def sparse_tsne_kmeans(data:numpy.ndarray, select_cnt:int, tsne_dim:int=2, seed=None) -> numpy.ndarray:
    """Furthest point sampling for data (shape [nsamples, nfeat])
    First apply tSNE, then apply KMeans
    """

    if seed is not None:
        numpy.random.seed(seed)
    tsne = TSNE(n_components=tsne_dim, init="pca", learning_rate="auto")
    data_tsne = tsne.fit_transform(data)
    kmeans = KMeans(n_clusters=select_cnt, random_state=seed)
    kmeans.fit(data_tsne)
    centroids = kmeans.cluster_centers_

    """get the index of each centroid in data"""
    dist = numpy.square(data_tsne.reshape(data_tsne.shape[0], 1, -1) - centroids.reshape(1, centroids.shape[0], -1)).sum(axis=-1)
    return numpy.argmin(dist, axis=0)


def sparse_random(data:numpy.ndarray, select_cnt:int, seed=None) -> numpy.ndarray:
    if not isinstance(seed, int):
        seed = 0
    numpy.random.seed(seed)
    idxes = numpy.arange(data.shape[0])
    numpy.random.shuffle(idxes)
    return idxes[:select_cnt]


def new_sample_method(data_path_dict:Dict[str, str], select_cnt:int, seed=None):
    """Try new sample methods for data (shape [nsamples, nfeat])
    """
    import inp
    from ase.io import read

    geoms = read(inp.filename, ":")

    """prepare species data"""
    atoms_species_flat = numpy.concatenate(numpy.array([g.get_chemical_symbols() for g in geoms]))
    atoms_species_uniq = numpy.sort(numpy.unique(atoms_species_flat))
    atoms_species_frac = {spe: numpy.sum(atoms_species_flat == spe)/len(atoms_species_flat) for spe in atoms_species_uniq}
    print(atoms_species_frac)
    species_idxes = {
        species: numpy.where(atoms_species_flat == species)[0]
        for species in atoms_species_uniq
    }
    assert set.union(*[set(v) for v in species_idxes.values()]) == set(range(len(atoms_species_flat)))

    assert set(atoms_species_uniq) == set(["Zr", "S"])

    assert inp.sample_method in ["fps", "kmeans", "tsne_kmeans", "random"]
    assert inp.sample_data_lam in [i for i in range(0, inp.nang1+1)]
    print(f"sample method: {inp.sample_method}, sample data lambda: {inp.sample_data_lam}")

    data = h5py.File(data_path_dict[inp.sample_data_lam], 'r')['descriptor'][:]
    data = data.reshape(data.shape[0] * data.shape[1], -1)

    """run case by case"""
    if inp.balance_species:
        select_cnt_balspe = {k: int(select_cnt*atoms_species_frac[k]) for k in atoms_species_uniq}
        if sum(select_cnt_balspe.values()) != select_cnt:
            select_cnt_balspe[atoms_species_uniq[0]] += select_cnt - sum(select_cnt_balspe.values())
        print(select_cnt_balspe)
        assert sum(select_cnt_balspe.values()) == select_cnt, \
            f"should have sum(select_cnt_balspe.values()) == select_cnt, but have {select_cnt=} and {select_cnt_balspe=}"
        if inp.sample_method == "fps":
            slct_idxes_species = {
                spe: do_fps_prepare_l(data[spe_idxes], select_cnt_balspe[spe], seed)
                for spe, spe_idxes in species_idxes.items()
            }
        elif inp.sample_method == "kmeans":
            slct_idxes_species = {
                spe: sparse_kmeans(data[spe_idxes], select_cnt_balspe[spe], seed)
                for spe, spe_idxes in species_idxes.items()
            }
        elif inp.sample_method == "tsne_kmeans":
            slct_idxes_species = {
                spe: sparse_tsne_kmeans(data[spe_idxes], select_cnt_balspe[spe], seed=seed)
                for spe, spe_idxes in species_idxes.items()
            }
        elif inp.sample_method == "random":
            slct_idxes_species = {
                spe: sparse_random(data[spe_idxes], select_cnt_balspe[spe], seed)
                for spe, spe_idxes in species_idxes.items()
            }
        else:
            raise ValueError(f"inp.sample_method should in ['fps', 'kmeans', 'tsne_kmeans'], but {inp.sample_method=}")

        slct_idxes_tf_species = {
            spe: numpy.zeros(len(spe_idxes), dtype=bool)
            for spe, spe_idxes in species_idxes.items()
        }
        for spe, slct_idxes_spe in slct_idxes_species.items():
            slct_idxes_tf_species[spe][slct_idxes_spe] = True
        slct_idxes_tf = numpy.zeros(data.shape[0], dtype=bool)
        for spe, spe_idxes in species_idxes.items():
            slct_idxes_tf[spe_idxes] = slct_idxes_tf_species[spe]
        slct_idxes = numpy.where(slct_idxes_tf)[0]
    else:
        if inp.sample_method == "fps":
            slct_idxes = do_fps_prepare_l(data, select_cnt, seed)
        elif inp.sample_method == "kmeans":
            slct_idxes = sparse_kmeans(data, select_cnt, seed)
        elif inp.sample_method == "tsne_kmeans":
            slct_idxes = sparse_tsne_kmeans(data, select_cnt, seed=seed)
        elif inp.sample_method == "random":
            slct_idxes = sparse_random(data, select_cnt, seed)
        else:
            raise ValueError(f"inp.sample_method should in ['fps', 'kmeans', 'tsne_kmeans'], but {inp.sample_method=}")

        slct_idxes_tf = numpy.zeros(data.shape[0], dtype=bool)  # TF array for all atoms
        slct_idxes_tf[slct_idxes] = True
        slct_idxes_tf_species = {  # TF array in each species
            spe: slct_idxes_tf[spe_idxes]
            for spe, spe_idxes in species_idxes.items()
        }
        slct_idxes_species = {  # selected index for each species (in each species array)
            spe: numpy.where(slct_idxes_tf_spe)[0]
            for spe, slct_idxes_tf_spe in slct_idxes_tf_species.items()
        }

    """finish up"""
    print("\n".join([f"{k:>3s} {len(v):>4d}" for k, v in slct_idxes_species.items()]))
    fig_pca = viz_distribution(data, slct_idxes, species_idxes, slct_idxes_species, method="pca")
    fig_pca.savefig(osp.join(inp.saltedpath, f"equirepr_{inp.saltedname}", f"viz_distribution_pca.png"))
    fig_tsne = viz_distribution(data, slct_idxes, species_idxes, slct_idxes_species, method="tsne")
    fig_tsne.savefig(osp.join(inp.saltedpath, f"equirepr_{inp.saltedname}", f"viz_distribution_tsne.png"))

    print(f"selected {len(slct_idxes)} from {data.shape[0]}")
    print(slct_idxes)
    return slct_idxes


def build():

    sys.path.insert(0, './')
    import inp

    species, lmax, nmax, llmax, nnmax, ndata, atomic_symbols, natoms, natmax = read_system()

    atom_idx, natom_dict = get_atom_idx(ndata,natoms,species,atomic_symbols)

    # number of sparse environments
    M = inp.Menv
    zeta = inp.z
    eigcut = inp.eigcut
    sdir = osp.join(inp.saltedpath, f"equirepr_{inp.saltedname}")

    if inp.field:
        kdir = osp.join(inp.saltedpath, f"kernels_{inp.saltedname}_field")
    else:
        kdir = osp.join(inp.saltedpath, f"kernels_{inp.saltedname}")

    # make directories if not exisiting
    if not osp.exists(kdir):
        os.mkdir(kdir)
    for spe in species:
        for l in range(llmax+1):
            dirpath = osp.join(kdir, f"spe{spe}_l{l}")
            if not osp.exists(dirpath):
                os.mkdir(dirpath)
            dirpath = osp.join(kdir, f"spe{spe}_l{l}", f"M{M}_zeta{zeta}")
            if not osp.exists(dirpath):
                os.mkdir(dirpath)

    def do_fps(x, d=0):
        # FPS code from Giulio Imbalzano
        if d == 0 : d = len(x)
        n = len(x)
        iy = np.zeros(d,int)
        iy[0] = 0
        # Faster evaluation of Euclidean distance
        n2 = np.sum((x*np.conj(x)),axis=1)
        dl = n2 + n2[iy[0]] - 2*np.real(np.dot(x,np.conj(x[iy[0]])))
        for i in range(1,d):
            iy[i] = np.argmax(dl)
            nd = n2 + n2[iy[i]] - 2*np.real(np.dot(x,np.conj(x[iy[i]])))
            dl = np.minimum(dl,nd)
        return iy

    # compute number of atomic environments for each species
    ispe = 0
    species_idx = {}
    for spe in species:
        species_idx[spe] = ispe
        ispe += 1

    species_array = np.zeros((ndata,natmax),int)
    for iconf in range(ndata):
        for iat in range(natoms[iconf]):
            spe = atomic_symbols[iconf][iat]
            species_array[iconf,iat] = species_idx[spe]
    species_array = species_array.reshape(ndata*natmax)

    # load lambda=0 power spectrum
    power = h5py.File(osp.join(sdir, "FEAT-0.h5"), 'r')['descriptor'][:]
    nfeat = power.shape[-1]

    ################################################################

    # # compute sparse set with FPS
    # fps_idx = np.array(do_fps(power.reshape(ndata*natmax,nfeat),M),int)

    try:
        inp.sample_method
    except AttributeError:
        print(f"inp.sample_method not found, use 'fps' as default")
        fps_idx = np.array(do_fps(power.reshape(ndata*natmax,nfeat),M),int)
    else:
        fps_idx = new_sample_method(
            {l:osp.join(sdir, f"FEAT-{l}.h5") for l in range(inp.nang1+1)},
            M,
        )

    ################################################################

    fps_species = species_array[fps_idx]
    sparse_set = np.vstack((fps_idx,fps_species)).T
    print("Computed sparse set made of ", M, "environments")
    np.savetxt(osp.join(sdir, f"sparse_set_{M}.txt"), sparse_set, fmt='%i')

    # divide sparse set per species
    fps_indexes = {}
    for spe in species:
        fps_indexes[spe] = []
    for iref in range(M):
        fps_indexes[species[fps_species[iref]]].append(fps_idx[iref])
    Mspe = {}
    for spe in species:
        Mspe[spe] = len(fps_indexes[spe])

    kernel0_mm = {}
    power_env_sparse = {}
    h5f = h5py.File(osp.join(sdir, f"FEAT-0-M-{M}.h5"), 'w')
    if inp.field:
        power2 = h5py.File(osp.join(sdir, "FEAT-0_field.h5"), 'r')['descriptor'][:]
        nfeat2 = power2.shape[-1]
        power_env_sparse2 = {}
        h5f2 = h5py.File(osp.join(sdir, f"FEAT-0-M-{M}_field.h5"), 'w')
    for spe in species:
        power_env_sparse[spe] = power.reshape(ndata*natmax,nfeat)[np.array(fps_indexes[spe],int)]
        h5f.create_dataset(spe,data=power_env_sparse[spe])
        if inp.field:
            power_env_sparse2[spe] = power2.reshape(ndata*natmax,nfeat2)[np.array(fps_indexes[spe],int)]
            h5f2.create_dataset(spe,data=power_env_sparse2[spe])
    h5f.close()
    if inp.field: h5f2.close()

    for spe in species:
        kernel0_mm[spe] = np.dot(power_env_sparse[spe],power_env_sparse[spe].T)
        if inp.field:
            kernel_mm = np.dot(power_env_sparse2[spe],power_env_sparse2[spe].T) * kernel0_mm[spe]**(zeta-1)
            #kernel_mm = np.dot(power_env_sparse2[spe],power_env_sparse2[spe].T) * np.exp(kernel0_mm[spe])
        else:
            kernel_mm = kernel0_mm[spe]**zeta

        eva, eve = np.linalg.eigh(kernel_mm)
        eva = eva[eva > eigcut]
        eve = eve[:,-len(eva):]
        V = np.dot(eve,np.diag(1.0/np.sqrt(eva)))
        np.save(osp.join(
            kdir, f"spe{spe}_l{0}", f"M{M}_zeta{zeta}", "projector.npy"
        ), V)

    for l in range(1,llmax+1):
        power = h5py.File(osp.join(sdir, f"FEAT-{l}.h5"), 'r')['descriptor'][:]
        nfeat = power.shape[-1]
        power_env_sparse = {}
        h5f = h5py.File(osp.join(sdir, f"FEAT-{l}-M-{M}.h5"), 'w')
        if inp.field:
            power2 = h5py.File(osp.join(sdir, f"FEAT-{l}_field.h5"),'r')['descriptor'][:]
            nfeat2 = power2.shape[-1]
            power_env_sparse2 = {}
            h5f2 = h5py.File(osp.join(sdir, f"FEAT-{l}-M-{M}_field.h5"),'w')
        for spe in species:
            power_env_sparse[spe] = power.reshape(ndata*natmax,2*l+1,nfeat)[np.array(fps_indexes[spe],int)].reshape(Mspe[spe]*(2*l+1),nfeat)
            h5f.create_dataset(spe,data=power_env_sparse[spe])
            if inp.field:
                power_env_sparse2[spe] = power2.reshape(ndata*natmax,2*l+1,nfeat2)[np.array(fps_indexes[spe],int)].reshape(Mspe[spe]*(2*l+1),nfeat2)
                h5f2.create_dataset(spe,data=power_env_sparse2[spe])
        h5f.close()
        if inp.field: h5f2.close()

        for spe in species:
            if inp.field:
                kernel_mm = np.dot(power_env_sparse2[spe],power_env_sparse2[spe].T)
            else:
                kernel_mm = np.dot(power_env_sparse[spe],power_env_sparse[spe].T)
            for i1 in range(Mspe[spe]):
                for i2 in range(Mspe[spe]):
                    kernel_mm[i1*(2*l+1):i1*(2*l+1)+2*l+1][:,i2*(2*l+1):i2*(2*l+1)+2*l+1] *= kernel0_mm[spe][i1,i2]**(zeta-1)
                    #kernel_mm[i1*(2*l+1):i1*(2*l+1)+2*l+1][:,i2*(2*l+1):i2*(2*l+1)+2*l+1] *= np.exp(kernel0_mm[spe][i1,i2])
            eva, eve = np.linalg.eigh(kernel_mm)
            eva = eva[eva>eigcut]
            eve = eve[:,-len(eva):]
            V = np.dot(eve,np.diag(1.0/np.sqrt(eva)))
            np.save(osp.join(
                kdir, f"spe{spe}_l{l}", f"M{M}_zeta{zeta}", "projector.npy"
            ), V)

    return

if __name__ == "__main__":
    build()

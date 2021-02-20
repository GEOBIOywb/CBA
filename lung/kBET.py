# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 10:41:54 2021

@author: 17b90
"""
import numpy as np
import anndata
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse.csgraph import connected_components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#import networkx as nx
from scIB.utils import *
from scIB.preprocessing import score_cell_cycle
from scIB.clustering import opt_louvain
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.io import mmwrite
import sklearn
import sklearn.metrics
from time import time
import cProfile
from pstats import Stats
import memory_profiler
import itertools
import multiprocessing as multipro
import subprocess
import tempfile
import pathlib
from os import mkdir, path, remove, stat
import gc
import rpy2.rinterface_lib.callbacks
import logging
rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR) # Ignore R warning messages
import rpy2.robjects as ro
import anndata2ri

def checkAdata(adata):
    if type(adata) is not anndata.AnnData:
        raise TypeError('Input is not a valid AnnData object')

def checkBatch(batch, obs, verbose=False):
    if batch not in obs:
        raise ValueError(f'column {batch} is not in obs')
    elif verbose:
        print(f'Object contains {obs[batch].nunique()} batches.')

def diffusion_conn(adata, min_k=50, copy=True, max_iterations=26):
    '''
    This function performs graph diffusion on the connectivities matrix until a
    minimum number `min_k` of entries per row are non-zero.
    
    Note:
    Due to self-loops min_k-1 non-zero connectivies entries is actually the stopping
    criterion. This is equivalent to `sc.pp.neighbors`.
    
    Returns:
       The diffusion-enhanced connectivities matrix of a copy of the AnnData object
       with the diffusion-enhanced connectivities matrix is in 
       `adata.uns["neighbors"]["conectivities"]`
    '''
    if 'neighbors' not in adata.uns:
        raise ValueError('`neighbors` not in adata object. '
                         'Please compute a neighbourhood graph!')

    if 'connectivities' not in adata.uns['neighbors']:
        raise ValueError('`connectivities` not in `adata.uns["neighbors"]`. '
                         'Please pass an object with connectivities computed!')


    T = adata.uns['neighbors']['connectivities']

    #Normalize T with max row sum
    # Note: This keeps the matrix symmetric and ensures |M| doesn't keep growing
    T = sparse.diags(1/np.array([T.sum(1).max()]*T.shape[0]))*T
    
    M = T

    # Check for disconnected component
    n_comp, labs = connected_components(adata.uns['neighbors']['connectivities'],
                                                       connection='strong')
    
    if n_comp > 1:
        tab = pd.value_counts(labs)
        small_comps = tab.index[tab<min_k]
        large_comp_mask = np.array(~pd.Series(labs).isin(small_comps))
    else:
        large_comp_mask = np.array([True]*M.shape[0])

    T_agg = T
    i = 2
    while ((M[large_comp_mask,:][:,large_comp_mask]>0).sum(1).min() < min_k) and (i < max_iterations):
        print(f'Adding diffusion to step {i}')
        T_agg *= T
        M += T_agg
        i+=1

    if (M[large_comp_mask,:][:,large_comp_mask]>0).sum(1).min() < min_k:
        raise ValueError('could not create diffusion connectivities matrix'
                         f'with at least {min_k} non-zero entries in'
                         f'{max_iterations} iterations.\n Please increase the'
                         'value of max_iterations or reduce k_min.\n')

    M.setdiag(0)

    if copy:
        adata_tmp = adata.copy()
        adata_tmp.uns['neighbors'].update({'diffusion_connectivities': M})
        return adata_tmp

    else:
        return M
    
def diffusion_nn(adata, k, max_iterations=26):
    '''
    This function generates a nearest neighbour list from a connectivities matrix
    as supplied by BBKNN or Conos. This allows us to select a consistent number
    of nearest neighbours across all methods.

    Return:
       `k_indices` a numpy.ndarray of the indices of the k-nearest neighbors.
    '''
    if 'neighbors' not in adata.uns:
        raise ValueError('`neighbors` not in adata object. '
                         'Please compute a neighbourhood graph!')
    
    if 'connectivities' not in adata.uns['neighbors']:
        raise ValueError('`connectivities` not in `adata.uns["neighbors"]`. '
                         'Please pass an object with connectivities computed!')
        
    T = adata.uns['neighbors']['connectivities']

    # Row-normalize T
    T = sparse.diags(1/T.sum(1).A.ravel())*T
    
    T_agg = T**3
    M = T+T**2+T_agg
    i = 4
    
    while ((M>0).sum(1).min() < (k+1)) and (i < max_iterations): 
        #note: k+1 is used as diag is non-zero (self-loops)
        print(f'Adding diffusion to step {i}')
        T_agg *= T
        M += T_agg
        i+=1

    if (M>0).sum(1).min() < (k+1):
        raise NeighborsError(f'could not find {k} nearest neighbors in {max_iterations}'
                         'diffusion steps.\n Please increase max_iterations or reduce'
                         ' k.\n')
    
    M.setdiag(0)
    k_indices = np.argpartition(M.A, -k, axis=1)[:, -k:]
    
    return k_indices

def kBET_single(matrix, batch, type_ = None, k0 = 10, knn=None, subsample=0.5, heuristic=True, verbose=False):
    """
    params:
        matrix: expression matrix (at the moment: a PCA matrix, so do.pca is set to FALSE
        batch: series or list of batch assignemnts
        subsample: fraction to be subsampled. No subsampling if `subsample=None`
    returns:
        kBET p-value
    """
        
    anndata2ri.activate()
    ro.r("library(kBET)")
    
    if verbose:
        print("importing expression matrix")
    ro.globalenv['data_mtrx'] = matrix
    ro.globalenv['batch'] = batch
    #print(matrix.shape)
    #print(len(batch))
    
    if verbose:
        print("kBET estimation")
    #k0 = len(batch) if len(batch) < 50 else 'NULL'
    
    ro.globalenv['knn_graph'] = knn
    ro.globalenv['k0'] = k0
    batch_estimate = ro.r(f"batch.estimate <- kBET(data_mtrx, batch, knn=knn_graph, k0=k0, plot=FALSE, do.pca=FALSE, heuristic=FALSE, adapt=FALSE, verbose={str(verbose).upper()})")
            
    anndata2ri.deactivate()
    try:
        ro.r("batch.estimate$average.pval")[0]
    except rpy2.rinterface_lib.embedded.RRuntimeError:
        return np.nan
    else:
        return ro.r("batch.estimate$average.pval")[0]
    
def kbet(adata, batch_key, label_key, embed='X_pca', type_ = None,
                    hvg=False, subsample=0.5, heuristic=False, verbose=False):
    """
    Compare the effect before and after integration
    params:
        matrix: matrix from adata to calculate on
    return:
        pd.DataFrame with kBET p-values per cluster for batch
    """
    
    checkAdata(adata)
    checkBatch(batch_key, adata.obs)
    checkBatch(label_key, adata.obs)
    #compute connectivities for non-knn type data integrations
    #and increase neighborhoods for knn type data integrations
    if type_ =='haveneighbor':
        adata_tmp = adata
        print('neighbor have already obtained!')
    elif type_ != 'knn':
        adata_tmp = sc.pp.neighbors(adata, n_neighbors = 50, use_rep=embed, copy=True)
    else:
        #check if pre-computed neighbours are stored in input file
        adata_tmp = adata.copy()
        if 'diffusion_connectivities' not in adata.uns['neighbors']:
            if verbose:
                print(f"Compute: Diffusion neighbours.")
            adata_tmp = diffusion_conn(adata, min_k = 50, copy = True)
        adata_tmp.uns['neighbors']['connectivities'] = adata_tmp.uns['neighbors']['diffusion_connectivities']
            
    if verbose:
        print(f"batch: {batch_key}")
        
    #set upper bound for k0
    size_max = 2**31 - 1
    
    kBET_scores = {'cluster': [], 'kBET': []}
    for clus in adata_tmp.obs[label_key].unique():
        
        adata_sub = adata_tmp[adata_tmp.obs[label_key] == clus,:].copy()
        #check if neighborhood size too small or only one batch in subset
        if np.logical_or(adata_sub.n_obs < 10, 
                         len(np.unique(adata_sub.obs[batch_key]))==1):
            print(f"{clus} consists of a single batch or is too small. Skip.")
            score = np.nan
        else:
            quarter_mean = np.floor(np.mean(adata_sub.obs[batch_key].value_counts())/4).astype('int')
            k0 = np.min([70, np.max([10, quarter_mean])])
            #check k0 for reasonability
            if (k0*adata_sub.n_obs) >=size_max:
                k0 = np.floor(size_max/adata_sub.n_obs).astype('int')
           
            matrix = np.zeros(shape=(adata_sub.n_obs, k0+1))
                
            if verbose:
                print(f"Use {k0} nearest neighbors.")
            n_comp, labs = connected_components(adata_sub.uns['neighbors']['connectivities'], 
                                                              connection='strong')
            if n_comp > 1:
                #check the number of components where kBET can be computed upon
                comp_size = pd.value_counts(labs)
                #check which components are small
                comp_size_thresh = 3*k0
                idx_nonan = np.flatnonzero(np.in1d(labs, 
                                                   comp_size[comp_size>=comp_size_thresh].index))
                #check if 75% of all cells can be used for kBET run
                if len(idx_nonan)/len(labs) >= 0.75:
                    #create another subset of components, assume they are not visited in a diffusion process
                    adata_sub_sub = adata_sub[idx_nonan,:].copy()
                    nn_index_tmp = np.empty(shape=(adata_sub.n_obs, k0))
                    nn_index_tmp[:] = np.nan
                    nn_index_tmp[idx_nonan] = diffusion_nn(adata_sub_sub, k=k0).astype('float') 
                    #need to check neighbors (k0 or k0-1) as input?   
                    score = kBET_single(
                            matrix=matrix,
                            batch=adata_sub.obs[batch_key],
                            knn = nn_index_tmp+1, #nn_index in python is 0-based and 1-based in R
                            subsample=subsample,
                            verbose=verbose,
                            heuristic=False,
                            k0 = k0,
                            type_ = type_
                            )
                else:
                    #if there are too many too small connected components, set kBET score to 1 
                    #(i.e. 100% rejection)
                    score = 1
                
            else: #a single component to compute kBET on 
                #need to check neighbors (k0 or k0-1) as input?  
                nn_index_tmp = diffusion_nn(adata_sub, k=k0).astype('float')
                score = kBET_single(
                            matrix=matrix,
                            batch=adata_sub.obs[batch_key],
                            knn = nn_index_tmp+1, #nn_index in python is 0-based and 1-based in R
                            subsample=subsample,
                            verbose=verbose,
                            heuristic=False,
                            k0 = k0,
                            type_ = type_
                            )
        
        kBET_scores['cluster'].append(clus)
        kBET_scores['kBET'].append(score)
    
    kBET_scores = pd.DataFrame.from_dict(kBET_scores)
    kBET_scores = kBET_scores.reset_index(drop=True)
    
    return kBET_scores


def nmi(adata, group1, group2, method="arithmetic", nmi_dir=None):
    """
    Normalized mutual information NMI based on 2 different cluster assignments `group1` and `group2`
    params:
        adata: Anndata object
        group1: column name of `adata.obs` or group assignment
        group2: column name of `adata.obs` or group assignment
        method: NMI implementation
            'max': scikit method with `average_method='max'`
            'min': scikit method with `average_method='min'`
            'geometric': scikit method with `average_method='geometric'`
            'arithmetic': scikit method with `average_method='arithmetic'`
            'Lancichinetti': implementation by A. Lancichinetti 2009 et al.
            'ONMI': implementation by Aaron F. McDaid et al. (https://github.com/aaronmcdaid/Overlapping-NMI) Hurley 2011
        nmi_dir: directory of compiled C code if 'Lancichinetti' or 'ONMI' are specified as `method`. Compilation should be done as specified in the corresponding README.
    return:
        normalized mutual information (NMI)
    """
    
    checkAdata(adata)
    
    if isinstance(group1, str):
        checkBatch(group1, adata.obs)
        group1 = adata.obs[group1].tolist()
    elif isinstance(group1, pd.Series):
        group1 = group1.tolist()
        
    if isinstance(group2, str):
        checkBatch(group2, adata.obs)
        group2 = adata.obs[group2].tolist()
    elif isinstance(group2, pd.Series):
        group2 = group2.tolist()
    
    if len(group1) != len(group2):
        raise ValueError(f'different lengths in group1 ({len(group1)}) and group2 ({len(group2)})')
    
    # choose method
    if method in ['max', 'min', 'geometric', 'arithmetic']:
        nmi_value = sklearn.metrics.normalized_mutual_info_score(group1, group2, average_method=method)
    elif method == "Lancichinetti":
        nmi_value = nmi_Lanc(group1, group2, nmi_dir=nmi_dir)
    elif method == "ONMI":
        nmi_value = onmi(group1, group2, nmi_dir=nmi_dir)
    else:
        raise ValueError(f"Method {method} not valid")
    
    return nmi_value


def ari(adata, group1, group2):
    """
    params:
        adata: anndata object
        group1: ground-truth cluster assignments (e.g. cell type labels)
        group2: "predicted" cluster assignments
    The function is symmetric, so group1 and group2 can be switched
    """
    
    checkAdata(adata)
    
    if isinstance(group1, str):
        checkBatch(group1, adata.obs)
        group1 = adata.obs[group1].tolist()
    elif isinstance(group1, pd.Series):
        group1 = group1.tolist()
        
    if isinstance(group2, str):
        checkBatch(group2, adata.obs)
        group2 = adata.obs[group2].tolist()
    elif isinstance(group2, pd.Series):
        group2 = group2.tolist()
    
    if len(group1) != len(group2):
        raise ValueError(f'different lengths in group1 ({len(group1)}) and group2 ({len(group2)})')
    
    return sklearn.metrics.cluster.adjusted_rand_score(group1, group2)

def silhouette(adata, group_key, metric='euclidean', embed='X_pca', scale=True):
    """
    wrapper for sklearn silhouette function values range from [-1, 1] with 1 being an ideal fit, 0 indicating overlapping clusters and -1 indicating misclassified cells
    """
    if embed not in adata.obsm.keys():
        print(adata.obsm.keys())
        raise KeyError(f'{embed} not in obsm')
    asw = sklearn.metrics.silhouette_score(adata.obsm[embed], adata.obs[group_key], metric=metric)
    if scale:
        asw = (asw + 1)/2
    return asw
### PC Regression        
def pcr_comparison(adata_pre, adata_post, covariate, embed=None, n_comps=50, scale=True, verbose=False):
    """
    Compare the effect before and after integration
    Return either the difference of variance contribution before and after integration
    or a score between 0 and 1 (`scaled=True`) with 0 if the variance contribution hasn't 
    changed. The larger the score, the more different the variance contributions are before 
    and after integration.
    params:
        adata_pre: uncorrected adata
        adata_post: integrated adata
        embed   : if `embed=None`, use the full expression matrix (`adata.X`), otherwise
                  use the embedding provided in `adata_post.obsm[embed]`
        scale: if True, return scaled score
    return:
        difference of R2Var value of PCR
    """
    
    if embed == 'X_pca':
        embed = None
    
    pcr_before = pcr(adata_pre, covariate=covariate, recompute_pca=True,
                     n_comps=n_comps, verbose=verbose)
    pcr_after = pcr(adata_post, covariate=covariate, embed=embed, recompute_pca=True,
                    n_comps=n_comps, verbose=verbose)

    if scale:
        score = (pcr_before - pcr_after)/pcr_before
        if score < 0:
            print("Variance contribution increased after integration!")
            print("Setting PCR comparison score to 0.")
            score = 0
        return score
    else:
        return pcr_after - pcr_before

def pcr(adata, covariate, embed=None, n_comps=50, recompute_pca=True, verbose=False):
    """
    PCR for Adata object
    Checks whether to
        + compute PCA on embedding or expression data (set `embed` to name of embedding matrix e.g. `embed='X_emb'`)
        + use existing PCA (only if PCA entry exists)
        + recompute PCA on expression matrix (default)
    params:
        adata: Anndata object
        embed   : if `embed=None`, use the full expression matrix (`adata.X`), otherwise
                  use the embedding provided in `adata_post.obsm[embed]`
        n_comps: number of PCs if PCA should be computed
        covariate: key for adata.obs column to regress against
    return:
        R2Var of PCR
    """
    
    checkAdata(adata)
    checkBatch(covariate, adata.obs)
    
    if verbose:
        print(f"covariate: {covariate}")
    batch = adata.obs[covariate]
    
    # use embedding for PCA
    if (embed is not None) and (embed in adata.obsm):
        if verbose:
            print(f"compute PCR on embedding n_comps: {n_comps}")
        return pc_regression(adata.obsm[embed], batch, n_comps=n_comps)
    
    # use existing PCA computation
    elif (recompute_pca == False) and ('X_pca' in adata.obsm) and ('pca' in adata.uns):
        if verbose:
            print("using existing PCA")
        return pc_regression(adata.obsm['X_pca'], batch, pca_var=adata.uns['pca']['variance'])
    
    # recompute PCA
    else:
        if verbose:
            print(f"compute PCA n_comps: {n_comps}")
        return pc_regression(adata.X, batch, n_comps=n_comps)

def pc_regression(data, variable, pca_var=None, n_comps=50, svd_solver='arpack', verbose=False):
    """
    params:
        data: expression or PCA matrix. Will be assumed to be PCA values, if pca_sd is given
        variable: series or list of batch assignments
        n_comps: number of PCA components for computing PCA, only when pca_sd is not given. If no pca_sd is given and n_comps=None, comute PCA and don't reduce data
        pca_var: iterable of variances for `n_comps` components. If `pca_sd` is not `None`, it is assumed that the matrix contains PCA values, else PCA is computed
    PCA is only computed, if variance contribution is not given (pca_sd).
    """

    if isinstance(data, (np.ndarray, sparse.csr_matrix)):
        matrix = data
    else:
        raise TypeError(f'invalid type: {data.__class__} is not a numpy array or sparse matrix')

    # perform PCA if no variance contributions are given
    if pca_var is None:

        if n_comps is None or n_comps > min(matrix.shape):
            n_comps = min(matrix.shape)

        if n_comps == min(matrix.shape):
            svd_solver = 'full'

        if verbose:
            print("compute PCA")
        pca = sc.tl.pca(matrix, n_comps=n_comps, use_highly_variable=False,
                        return_info=True, svd_solver=svd_solver, copy=True)
        X_pca = pca[0].copy()
        pca_var = pca[3].copy()
        del pca
    else:
        X_pca = matrix
        n_comps = matrix.shape[1]

    ## PC Regression
    if verbose:
        print("fit regression on PCs")

    # handle categorical values
    if pd.api.types.is_numeric_dtype(variable):
        variable = np.array(variable).reshape(-1, 1)
    else:
        if verbose:
            print("one-hot encode categorical values")
        variable = pd.get_dummies(variable)

    # fit linear model for n_comps PCs
    r2 = []
    for i in range(n_comps):
        pc = X_pca[:, [i]]
        lm = sklearn.linear_model.LinearRegression()
        lm.fit(variable, pc)
        r2_score = np.maximum(0,lm.score(variable, pc))
        r2.append(r2_score)

    Var = pca_var / sum(pca_var) * 100
    R2Var = sum(r2*Var)/100

    return R2Var
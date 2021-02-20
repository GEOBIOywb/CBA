# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 18:58:59 2020

@author: 17b90
"""
import keras as K
import pandas as pd
from keras import layers
import numpy as np
import matplotlib.pyplot as plt 
import tensorflow as tf
from sklearn.decomposition import PCA
import scanpy as sc
import scipy
import pickle
from sklearn.manifold import TSNE
from keras.layers.core import Lambda
import scipy.io as sio
import seaborn as sns
import umap
import numpy as np
import metrics
from ywb_function import *
import scanorama
import sklearn.metrics as sm
import kBET

we_use=[1,2]#we try to integrate pancreas1 and pancreas2

#input the data
RAWseries1=pd.read_csv('RAWseries_'+str(we_use[0])+'.csv',header=None)[1:].values.astype('single')
RAWseries2=pd.read_csv('RAWseries_'+str(we_use[1])+'.csv',header=None)[1:].values.astype('single')
#input the label
choose_seriestype1=pd.read_csv('realseries_'+str(we_use[0])+'.csv',header=None)[1:].values
choose_seriestype2=pd.read_csv('realseries_'+str(we_use[1])+'.csv',header=None)[1:].values

Alldata=np.concatenate([RAWseries1.T,RAWseries2.T])
Alllabel=np.concatenate([choose_seriestype1,choose_seriestype2])
Allbatch=np.concatenate([np.zeros(choose_seriestype1.shape[0]),np.zeros(choose_seriestype2.shape[0])+1])
###############################################################################
chosen_cluster=['alpha','beta','ductal','acinar','delta','gamma','endothelial','epsilon']
chosen_index=np.arange(Alllabel.shape[0])
for i in range(Alllabel.shape[0]):
    if Alllabel[i] in chosen_cluster:
        chosen_index[i]=1
    else:
        chosen_index[i]=0
Alldata=Alldata[chosen_index==1,:]
Allbatch=Allbatch[chosen_index==1]
Alllabel=Alllabel[chosen_index==1]
###############################################################################
Numlabel=np.zeros(Alllabel.shape[0])
cluster_index2={'alpha':0,'beta':1,'ductal':2,'acinar':3,'delta':4,'gamma':5,'endothelial':6,'epsilon':7}
for i in range(Alllabel.shape[0]):
    Numlabel[i]=cluster_index2[Alllabel[i][0]]
###############################################################################
choose_seriestype1=Numlabel[Allbatch==0][Numlabel[Allbatch==0].argsort()].astype('int')
choose_seriestype2=Numlabel[Allbatch==1][Numlabel[Allbatch==1].argsort()].astype('int')
Numlabel[Allbatch==0]=choose_seriestype1
Numlabel[Allbatch==1]=choose_seriestype2
total_given_type=Numlabel

merge=sio.loadmat('pancreas_ourdata')['mergedata']

#here is hard, you need to check which one is batch1 and which one is batch2, I do that manually
mergedata=sc.AnnData(merge)
total_batch_type=np.concatenate([choose_seriestype1*0,choose_seriestype2*0+1])
total_batch_type=np.reshape(total_batch_type,total_batch_type.shape[0])
mergedata.obs['batch']=total_batch_type

zero_type=np.concatenate([choose_seriestype1*0,choose_seriestype2*0])
zero_type=np.reshape(zero_type,zero_type.shape[0])
mergedata.obs['zero']=zero_type

total_given_type=np.concatenate([choose_seriestype1,choose_seriestype2])
total_given_type=np.reshape(total_given_type,total_given_type.shape[0])
mergedata.obs['real']=total_given_type

mergedata.obsm["embedding"]=mergedata.X

sc.pp.neighbors(mergedata,n_pcs=0)
mergedata.obsm['NN']=mergedata.X
sc.tl.louvain(mergedata,resolution=0.5)
sc.tl.umap(mergedata)
sc.pl.umap(mergedata,color=['batch','louvain','real'])

type_louvain=mergedata.obs['louvain']
type_batch=mergedata.obs['batch']
type_real=mergedata.obs['real']
type_batch=type_batch.replace('ref',0)
type_batch=type_batch.replace('new',1)
###############################################################################
kBET_score=kBET.kbet(mergedata,'batch','real',embed='embedding')
for i in range(8):
    print(kBET_score['kBET'][i])
print(kBET_score.mean()[1])

kBET_score_whole=kBET.kbet(mergedata,'batch','zero',embed='embedding')
print(kBET_score_whole.mean()[1])

S_score=kBET.silhouette(mergedata,'real',metric='euclidean',embed='embedding')
print(S_score)

NMI_louvain=kBET.nmi(mergedata,'louvain','real')
print(NMI_louvain)

ARI_louvain=kBET.ari(mergedata,'louvain','real')
print(ARI_louvain)

FMI_louvain=sm.fowlkes_mallows_score(type_real,type_louvain)
print(FMI_louvain)
###############################################################################
umapdata=pd.DataFrame(mergedata.obsm['X_umap'].T,index=['tSNE1','tSNE2'])
umapdata1=pd.DataFrame(mergedata.obsm['X_umap'][0:(Allbatch==0).sum(),:].T,index=['tSNE1','tSNE2'])
umapdata2=pd.DataFrame(mergedata.obsm['X_umap'][(Allbatch==0).sum():,:].T,index=['tSNE1','tSNE2'])
##############################################################################
fromname='do'
plot_tSNE_clusters(umapdata,list(map(int,type_real)), cluster_colors=cluster_colors,save=False, name=fromname+'label')
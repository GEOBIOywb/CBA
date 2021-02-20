"""
Created on Fri Mar 27 18:58:59 2020

@author: 17b90
"""
import kBET
import scipy
import random
import keras as K
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import scipy.io as sio
import tensorflow as tf
from keras import layers
from ywb_function import *
import sklearn.metrics as sm
from collections import Counter
import matplotlib.pyplot as plt 
from keras.regularizers import l2
from sklearn import preprocessing
from keras.layers.core import Lambda
from keras.callbacks import TensorBoard
from imblearn.over_sampling import SMOTE,ADASYN
from keras.callbacks import LearningRateScheduler
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import dendrogram, linkage
#input the data
H_acc=sc.read_mtx('GSE127774_ACC_H_matrix.mtx')
C_acc=sc.read_mtx('GSE127774_ACC_C_matrix.mtx')

H_acc_data=scipy.sparse.csr_matrix(H_acc.X, dtype=np.int8).toarray()
C_acc_data=scipy.sparse.csr_matrix(C_acc.X, dtype=np.int8).toarray()

H_acc_gene=pd.read_csv('GSE127774_ACC_H_genes.csv', header=None)
H_acc_data=pd.DataFrame(data=H_acc_data, index=H_acc_gene[0].values).astype(float)
C_acc_gene=pd.read_csv('GSE127774_ACC_C_genes.csv', header=None)
C_acc_data=pd.DataFrame(data=C_acc_data, index=C_acc_gene[0].values).astype(float)

human_chimpanzee_genecouple=pd.read_csv('human_chimpanzee.csv', header=None)

row=[]
for i in range(human_chimpanzee_genecouple.shape[0]):
    if (human_chimpanzee_genecouple.values==human_chimpanzee_genecouple.loc[i][0]).sum()>=2 or (human_chimpanzee_genecouple.values==human_chimpanzee_genecouple.loc[i][1]).sum()>=2:
        human_chimpanzee_genecouple.loc[i][0]='0'
        human_chimpanzee_genecouple.loc[i][1]='0'
        row.append(i)
human_chimpanzee_genecouple_new=human_chimpanzee_genecouple.drop(row)
human_chimpanzee_genecouple_new=pd.DataFrame(human_chimpanzee_genecouple_new.values)

series1=human_expressionlevel
series2=chimpanzee_expressionlevel
gene_couple=human_chimpanzee_genecouple_new

series1_gene=gene_couple[0][1:].values
series2_gene=gene_couple[1][1:].values

#to remove genes which only exist in single species
series1_gene='hg38____'+series1_gene
series2_gene='panTro5_'+series2_gene

series1_gene=list(series1_gene)
series2_gene=list(series2_gene)

for i in range(len(series1_gene)):
    if series1_gene[i] not in list(series1.index) or series2_gene[i] not in list(series2.index):
        series1_gene[i]=0
        series2_gene[i]=0
series1_gene=list(filter(lambda x:x!=0,series1_gene))
series2_gene=list(filter(lambda x:x!=0,series2_gene))

#only choose these genes
series1_choose=series1.loc[series1_gene]
series2_choose=series2.loc[series2_gene]

series1_ann=sc.AnnData((series1_choose.values).T,obs=pd.DataFrame(series1_choose.columns), var=pd.DataFrame(series1_choose.index))
series2_ann=sc.AnnData((series2_choose.values).T,obs=pd.DataFrame(series2_choose.columns), var=pd.DataFrame(series2_choose.index))

RAWseries1=series1_ann.X.T
RAWseries2=series2_ann.X.T

fromname='humanchimpanzee'
pca_dim=20#the number of PCs
clusternumber=1
###############################################################################
anndata1=sc.AnnData(RAWseries1.T)
celluse=np.arange(0,anndata1.shape[0])
anndata1.obs['usecell']=celluse
sc.pp.filter_cells(anndata1,min_genes=20)#we cant to select some human cells, because my laptop is not good, so many cells are hard for it to do the training, moreover, the memory is also not enough

anndata2=sc.AnnData(RAWseries2.T)
celluse=np.arange(0,anndata2.shape[0])
anndata2.obs['usecell']=celluse
sc.pp.filter_cells(anndata2,min_genes=20)

anndata=anndata1.concatenate(anndata2)
sc.pp.filter_genes(anndata,min_cells=50)

sc.pp.normalize_per_cell(anndata,counts_per_cell_after=1e4)
sc.pp.log1p(anndata)

sc.pp.highly_variable_genes(anndata)
sc.pl.highly_variable_genes(anndata)

anndata=anndata[:,anndata.var['highly_variable']]

sc.tl.pca(anndata,n_comps=pca_dim)

Obtainseries1=(anndata.obsm['X_pca'])[anndata.obs['batch']=='0',:]
Obtainseries2=(anndata.obsm['X_pca'])[anndata.obs['batch']=='1',:]

Obtainseries1=sc.AnnData(Obtainseries1)
Obtainseries2=sc.AnnData(Obtainseries2)

sc.pp.neighbors(Obtainseries1,n_pcs=0)
sc.tl.umap(Obtainseries1)
sc.tl.louvain(Obtainseries1,resolution=1)
sc.pl.umap(Obtainseries1,color='louvain',size=30)

sc.pp.neighbors(Obtainseries2,n_pcs=0)
sc.tl.umap(Obtainseries2)
sc.tl.louvain(Obtainseries2,resolution=1)
sc.pl.umap(Obtainseries2,color='louvain',size=30)

PCAseries1=Obtainseries1.X
PCAseries2=Obtainseries2.X
###############################################################################
recluster1=np.array(list(map(int,Obtainseries1.obs['louvain'])))
recluster2=np.array(list(map(int,Obtainseries2.obs['louvain'])))
###############################################################################
#for i in range(len(np.unique(recluster1))):
#    print((np.where(recluster1==i))[0].shape[0])
#for i in range(len(np.unique(recluster2))):
#    print((np.where(recluster2==i))[0].shape[0])
#
##for the first batch
#number_cluster1=len(np.unique(recluster1))
#series1_data=np.zeros([0,PCAseries1.shape[1]])
#series1_index=np.zeros([0])
#recluster1plus=np.zeros([0])
#alpha=3#because the limiattion of memory of my laptop, I have to retain 1/3 human cells,so I preserve 1/3 human cells in each louvain cluster, this step is also unsupervised
#for i in range(number_cluster1):
#    index=np.where(recluster1==i)[0]
#    random.shuffle(index)
#    series1_data=np.concatenate([series1_data,(PCAseries1)[index[0::alpha]]])
#    series1_index=np.concatenate([series1_index,index[0::alpha]])
#    recluster1plus=np.concatenate([recluster1plus,np.zeros([index[0::alpha].shape[0]])+i])
#
##for the second batch
#number_cluster2=len(np.unique(recluster2))
#series2_data=np.zeros([0,PCAseries2.shape[1]])
#series2_index=np.zeros([0])
#recluster2plus=np.zeros([0])
#beta=1#fortunately, we could retain all chimp cells!!!!!
#for i in range(number_cluster2):
#    index=np.where(recluster2==i)[0]
#    random.shuffle(index)
#    series2_data=np.concatenate([series2_data,(PCAseries2)[index[0::beta]]])
#    series2_index=np.concatenate([series2_index,index[0::beta]])
#    recluster2plus=np.concatenate([recluster2plus,np.zeros([index[0::beta].shape[0]])+i])
#
#sio.savemat('series1_index.mat',{'series1_index':series1_index})
#sio.savemat('series2_index.mat',{'series2_index':series2_index})

#this is the indexes of cells I used
series1_index=sio.loadmat('series1_index.mat')['series1_index'][0].astype('int')
series2_index=sio.loadmat('series2_index.mat')['series2_index'][0].astype('int')

PCAseries1=PCAseries1[series1_index]
PCAseries2=PCAseries2[series2_index]
recluster1=recluster1[series1_index]
recluster2=recluster2[series2_index]

recluster1=recluster1.astype('int')
recluster2=recluster2.astype('int')
print(recluster1.shape[0])
print(recluster2.shape[0])
###############################################################################
def dis(P,Q,distance_method):
    if distance_method==0:
        return np.sqrt(np.sum(np.square(P-Q)))
    if distance_method==1:
        return 1-(np.multiply(P,Q).sum()/(np.sqrt(np.sum(np.square(P)))*np.sqrt(np.sum(np.square(Q)))))
###############################################################################
change=0
if len(np.unique(recluster1))<=len(np.unique(recluster2)):
    PCAseries1plus=PCAseries2
    PCAseries2plus=PCAseries1
    
    recluster1plus=recluster2
    recluster2plus=recluster1
    change=1
else:
    PCAseries1plus=PCAseries1
    PCAseries2plus=PCAseries2

    recluster1plus=recluster1
    recluster2plus=recluster2
###############################################################################
#ok, let's calculate the similarity of cells/clusters
correlation_recluster=np.zeros([len(np.unique(recluster1plus)),len(np.unique(recluster2plus))])
correlation_recluster_cell=np.zeros([recluster1plus.shape[0],recluster2plus.shape[0]])

for i in range(len(np.unique(recluster1plus))):
    for j in range(len(np.unique(recluster2plus))):
        print(i,j)
        index_series1=np.where(recluster1plus==i)[0]
        index_series2=np.where(recluster2plus==j)[0]

        cell_series1=PCAseries1plus[index_series1,:]
        cell_series2=PCAseries2plus[index_series2,:]

        mean1=0
        for iq in range(cell_series1.shape[0]):
            for jq in range(cell_series2.shape[0]):
                mean1+=dis(cell_series1[iq,:],cell_series2[jq,:],1)
        correlation_recluster[i,j]=mean1/(cell_series1.shape[0]*cell_series2.shape[0])
        
        for ii in range(cell_series1.shape[0]):
            for jj in range(cell_series2.shape[0]):
                mean2=dis(cell_series1[ii,:],cell_series2[jj,:],0)
                correlation_recluster_cell[index_series1[ii],index_series2[jj]]=mean2+0.00001

plt.imshow(correlation_recluster)
plt.imshow(correlation_recluster_cell)

correlation_recluster_div=-np.log10(correlation_recluster)
correlation_recluster_cell_div=-np.log10(correlation_recluster_cell)

correlation_recluster_norm=(correlation_recluster_div-correlation_recluster_div.min())/(correlation_recluster_div.max()-correlation_recluster_div.min())
correlation_recluster_cell_norm=(correlation_recluster_cell_div-correlation_recluster_cell_div.min())/(correlation_recluster_cell_div.max()-correlation_recluster_cell_div.min())

plt.imshow(correlation_recluster_norm)
plt.imshow(correlation_recluster_cell_norm)
###############################################################################
#remove bad parts, do the matching
correlation_recluster_select=np.zeros(correlation_recluster_norm.shape)

recluster_mid=np.zeros(recluster1plus.shape)
for kk in range(correlation_recluster_norm.shape[0]):
    ind=np.sort(correlation_recluster_norm[kk,:])
    select=correlation_recluster_norm[kk,:]<ind[-clusternumber]
    select=(select==False)
    recluster_mid[recluster1plus==kk]+=int(np.where(select==True)[0])
    correlation_recluster_select[kk,:]=correlation_recluster_norm[kk,:]*select
plt.imshow(correlation_recluster_select)

correlation_recluster_cell_final=correlation_recluster_cell*0
for i in range(correlation_recluster_cell_norm.shape[0]):
    for j in range(correlation_recluster_cell_norm.shape[1]):
        label1=recluster1plus[i]
        label2=recluster2plus[j]
        mean1=correlation_recluster_select[label1,label2]
        mean2=correlation_recluster_cell_norm[i,j]
        if mean1==0:
            correlation_recluster_cell_final[i,j]=0
        else:
            correlation_recluster_cell_final[i,j]=mean2
 
plt.imshow(correlation_recluster_select)
plt.imshow(correlation_recluster_cell_final)

recluster1plus=recluster_mid.astype('int')

np.unique(recluster1plus)
np.unique(recluster2plus)

sort_correlation_recluster_cell_final=correlation_recluster_cell_final[recluster1plus.argsort(),:]
sort_correlation_recluster_cell_final=sort_correlation_recluster_cell_final[:,recluster2plus.argsort()]

heatmap(sort_correlation_recluster_cell_final,recluster1plus,recluster2plus,save=True,name='speciesmatrix')
###############################################################################
if change==1:
    PCAseries1=PCAseries2plus
    PCAseries2=PCAseries1plus
    
    recluster1=recluster2plus
    recluster2=recluster1plus
else:
    PCAseries1=PCAseries1plus
    PCAseries2=PCAseries2plus
    
    recluster1=recluster1plus
    recluster2=recluster2plus
###############################################################################    
Obtainseries1plus=sc.AnnData(PCAseries1)
Obtainseries2plus=sc.AnnData(PCAseries2)

sc.pp.neighbors(Obtainseries1plus,n_pcs=0)
sc.tl.umap(Obtainseries1plus)

df=pd.DataFrame(recluster1.astype('int'))
df=pd.Series(np.reshape(df.values,df.values.shape[0]), dtype="category")
Obtainseries1plus.obs['louvain']=df.values

sc.pl.umap(Obtainseries1plus,color='louvain',size=30)
umapdata1=pd.DataFrame(Obtainseries1plus.obsm['X_umap'].T,
                       index=['tSNE1','tSNE2'])

plot_tSNE_clusters(umapdata1,Obtainseries1plus.obs['louvain'],cluster_colors=cluster_colors,save=False, name=fromname+'louvain')

sc.pp.neighbors(Obtainseries2plus,n_pcs=0)
sc.tl.umap(Obtainseries2plus)

df=pd.DataFrame(recluster2.astype('int'))
df=pd.Series(np.reshape(df.values,df.values.shape[0]), dtype="category")
Obtainseries2plus.obs['louvain']=df.values

sc.pl.umap(Obtainseries2plus,color='louvain',size=30)
umapdata2=pd.DataFrame(Obtainseries2plus.obsm['X_umap'].T,
                       index=['tSNE1','tSNE2'])

plot_tSNE_clusters(umapdata2,Obtainseries2plus.obs['louvain'],cluster_colors=cluster_colors,save=False, name=fromname+'louvain')
###############################################################################
#ok, I use keras, cells in each input are randomly selected, I don't know how to match cells with their similarity
#I also don't know how to match the cell part with their distance, so I design the following inputs
#It will waste some time, it's not easy and unclear for readers, but it works! 
PCAseries=np.concatenate([PCAseries1,PCAseries2])
PCAseries=preprocessing.StandardScaler().fit_transform(PCAseries)
PCAseries=preprocessing.MinMaxScaler().fit_transform(PCAseries)
PCAseries1=PCAseries[0:PCAseries1.shape[0]]
PCAseries2=PCAseries[PCAseries1.shape[0]:]

x_input1=np.zeros([PCAseries1.shape[0],PCAseries1.shape[1]+PCAseries1.shape[0]+PCAseries2.shape[0]+max(recluster1.max(),recluster2.max())+1])
x_input2=np.zeros([PCAseries2.shape[0],PCAseries2.shape[1]+PCAseries2.shape[0]+PCAseries1.shape[0]+max(recluster1.max(),recluster2.max())+1])
for i in range(PCAseries1.shape[0]):
    print(i)
    x_input1[i,0:PCAseries1.shape[1]]=PCAseries1[i,:]
    x_input1[i,PCAseries1.shape[1]:PCAseries1.shape[1]+PCAseries1.shape[0]]=K.utils.np_utils.to_categorical(i,PCAseries1.shape[0])
    x_input1[i,PCAseries1.shape[1]+PCAseries1.shape[0]:PCAseries1.shape[1]+PCAseries1.shape[0]+PCAseries2.shape[0]]=correlation_recluster_cell_final[i,:]
    x_input1[i,PCAseries1.shape[1]+PCAseries1.shape[0]+PCAseries2.shape[0]:]=K.utils.np_utils.to_categorical(recluster1[i],max(recluster1.max(),recluster2.max())+1)
    
for j in range(PCAseries2.shape[0]):
    print(j)
    x_input2[j,0:PCAseries2.shape[1]]=PCAseries2[j,:]
    x_input2[j,PCAseries2.shape[1]:PCAseries2.shape[1]+PCAseries2.shape[0]]=K.utils.np_utils.to_categorical(j,PCAseries2.shape[0])
    x_input2[j,PCAseries2.shape[1]+PCAseries2.shape[0]:PCAseries2.shape[1]+PCAseries2.shape[0]+PCAseries1.shape[0]]=correlation_recluster_cell_final[:,j]
    x_input2[j,PCAseries2.shape[1]+PCAseries2.shape[0]+PCAseries1.shape[0]:]=K.utils.np_utils.to_categorical(recluster2[j],max(recluster1.max(),recluster2.max())+1)
###############################################################################
#interesting, I need to make two batches have the same number of cells, so I have to copy cells again and again
if x_input1.shape[0]>=x_input2.shape[0]:
    x_test1=x_input1
    y_test1=recluster1
    y_testreal1=choose_seriestype1
    repeat_num=int(np.ceil(x_input1.shape[0]/x_input2.shape[0]))
    x_test2=np.tile(x_input2,(repeat_num,1))
    y_test2=np.tile(recluster2,repeat_num)
    y_testreal2=np.tile(choose_seriestype2,repeat_num)
    x_test2=x_test2[0:x_test1.shape[0],:]
    y_test2=y_test2[0:x_test1.shape[0]]
    y_testreal2=y_testreal2[0:x_test1.shape[0]]
elif x_input1.shape[0]<x_input2.shape[0]:
    x_test2=x_input2
    y_test2=recluster2       
    y_testreal2=choose_seriestype2
    repeat_num=int(np.ceil(x_input2.shape[0]/x_input1.shape[0])) 
    x_test1=np.tile(x_input1,(repeat_num,1))
    y_test1=np.tile(recluster1,repeat_num)
    y_testreal1=np.tile(choose_seriestype1,repeat_num)
    x_test1=x_test1[0:x_test2.shape[0],:]
    y_test1=y_test1[0:x_test2.shape[0]]
    y_testreal1=y_testreal1[0:x_test2.shape[0]]
###############################################################################
def choose_info(x,info_number):
    return x[:,0:info_number]
def choose_index(x,info_number,x_samplenumber):
    return x[:,info_number:info_number+x_samplenumber]
def choose_corrlation(x,info_number,x_samplenumber,cor_number):
    return x[:,info_number+x_samplenumber:info_number+x_samplenumber+cor_number]
def choose_relabel(x,info_number,x_samplenumber,cor_number):
    return x[:,info_number+x_samplenumber+cor_number:]
def slic(input_):
    return input_[:,0]
###############################################################################
activation='relu'
info_number=PCAseries1.shape[1]
layer=PCAseries1.shape[1]

input1=K.Input(shape=(x_test1.shape[1],))#line1 species1
input2=K.Input(shape=(x_test2.shape[1],))#line1 species2 
input3=K.Input(shape=(x_test1.shape[1],))#line2 species1
input4=K.Input(shape=(x_test2.shape[1],))#line2 species2 

Data1=Lambda(choose_info,arguments={'info_number':info_number})(input1)
Data2=Lambda(choose_info,arguments={'info_number':info_number})(input2)
Data3=Lambda(choose_info,arguments={'info_number':info_number})(input3)
Data4=Lambda(choose_info,arguments={'info_number':info_number})(input4)

Index1=Lambda(choose_index,arguments={'info_number':info_number,'x_samplenumber':PCAseries1.shape[0]})(input1)
Index2=Lambda(choose_index,arguments={'info_number':info_number,'x_samplenumber':PCAseries2.shape[0]})(input2)
Index3=Lambda(choose_index,arguments={'info_number':info_number,'x_samplenumber':PCAseries1.shape[0]})(input3)
Index4=Lambda(choose_index,arguments={'info_number':info_number,'x_samplenumber':PCAseries2.shape[0]})(input4)

Cor1=Lambda(choose_corrlation,arguments={'info_number':info_number,'x_samplenumber':PCAseries1.shape[0],'cor_number':PCAseries2.shape[0]})(input1)
Cor2=Lambda(choose_corrlation,arguments={'info_number':info_number,'x_samplenumber':PCAseries2.shape[0],'cor_number':PCAseries1.shape[0]})(input2)
Cor3=Lambda(choose_corrlation,arguments={'info_number':info_number,'x_samplenumber':PCAseries1.shape[0],'cor_number':PCAseries2.shape[0]})(input3)
Cor4=Lambda(choose_corrlation,arguments={'info_number':info_number,'x_samplenumber':PCAseries2.shape[0],'cor_number':PCAseries1.shape[0]})(input4)

Relabel1=Lambda(choose_relabel,arguments={'info_number':info_number,'x_samplenumber':PCAseries1.shape[0],'cor_number':PCAseries2.shape[0]})(input1)
Relabel2=Lambda(choose_relabel,arguments={'info_number':info_number,'x_samplenumber':PCAseries2.shape[0],'cor_number':PCAseries1.shape[0]})(input2)
Relabel3=Lambda(choose_relabel,arguments={'info_number':info_number,'x_samplenumber':PCAseries1.shape[0],'cor_number':PCAseries2.shape[0]})(input3)
Relabel4=Lambda(choose_relabel,arguments={'info_number':info_number,'x_samplenumber':PCAseries2.shape[0],'cor_number':PCAseries1.shape[0]})(input4)

x_concat1=layers.concatenate([Data1,Data3])#batch1
x_concat2=layers.concatenate([Data2,Data4])#batch2

x1=layers.Dense(layer,activation=activation)(Data1)
x2=layers.Dense(layer,activation=activation)(Data2)
x3=layers.Dense(layer,activation=activation)(Data3)
x4=layers.Dense(layer,activation=activation)(Data4)

x1=layers.BatchNormalization()(x1)
x2=layers.BatchNormalization()(x2)
x3=layers.BatchNormalization()(x3)
x4=layers.BatchNormalization()(x4)

x1_mid1=layers.Dense(layer,activation=activation)(layers.concatenate([x1,x2]))
x2_mid1=layers.Dense(layer,activation=activation)(layers.concatenate([x1,x2]))
x1_mid2=layers.Dense(layer,activation=activation)(layers.concatenate([x3,x4]))
x2_mid2=layers.Dense(layer,activation=activation)(layers.concatenate([x3,x4]))

x1_mid1=layers.BatchNormalization()(x1_mid1)
x2_mid1=layers.BatchNormalization()(x2_mid1)
x1_mid2=layers.BatchNormalization()(x1_mid2)
x2_mid2=layers.BatchNormalization()(x2_mid2)

layer_classify=layers.Dense(max(recluster1.max(),recluster2.max())+1,activation='relu')

y1=layer_classify(x1_mid1)
y2=layer_classify(x2_mid1)
y3=layer_classify(x1_mid2)
y4=layer_classify(x2_mid2)

x1=layers.concatenate([x1_mid1,x1_mid2])#batch1
x2=layers.concatenate([x2_mid1,x2_mid2])#batch2

output1=layers.Dense(2*layer,activation=activation)(x1)
output2=layers.Dense(2*layer,activation=activation)(x2)

output1=layers.BatchNormalization()(output1)
output2=layers.BatchNormalization()(output2)

def loss_weight(input_):
    return tf.reduce_sum(tf.multiply(input_[0],input_[1]),axis=-1)
def MSE(input_):
    return tf.reduce_mean(tf.square(input_[0]-input_[1]),axis=-1)
def multi_classification_loss(input_):
    return tf.keras.losses.categorical_crossentropy(input_[0],input_[1])

AE_loss_1=Lambda(MSE)([output1,x_concat1])
AE_loss_2=Lambda(MSE)([output2,x_concat2])

cls_loss_1=Lambda(MSE)([y1,Relabel1])
cls_loss_2=Lambda(MSE)([y2,Relabel2])
cls_loss_3=Lambda(MSE)([y3,Relabel3])
cls_loss_4=Lambda(MSE)([y4,Relabel4])

interweight1=Lambda(loss_weight)([Index1,Cor2])
interweight4=Lambda(loss_weight)([Index3,Cor4])

interloss_1=Lambda(MSE)([x1_mid1,x2_mid1])
interloss_4=Lambda(MSE)([x1_mid2,x2_mid2])

interloss_1=layers.Multiply()([interweight1,interloss_1])
interloss_4=layers.Multiply()([interweight4,interloss_4])

intraweight1=Lambda(loss_weight)([Relabel1,Relabel3])
intraweight2=Lambda(loss_weight)([Relabel2,Relabel4])

intraloss_1=Lambda(MSE)([x1_mid1,x1_mid2])
intraloss_2=Lambda(MSE)([x2_mid1,x2_mid2])

intraloss_1=layers.Multiply()([intraweight1,intraloss_1])
intraloss_2=layers.Multiply()([intraweight2,intraloss_2])

Loss1=Lambda(lambda x:(x[0]*1+x[1]*1)/2,name='loss1')([AE_loss_1,AE_loss_2])
Loss2=Lambda(lambda x:(x[0]*1+x[1]*1+x[2]*1+x[3]*1)/4,name='loss2')([cls_loss_1,cls_loss_2,cls_loss_3,cls_loss_4])
Loss3=Lambda(lambda x:(x[0]*1+x[1]*1)/2,name='loss3')([interloss_1,interloss_4])
Loss4=Lambda(lambda x:(x[0]*1+x[1]*1)/2,name='loss4')([intraloss_1,intraloss_2])
###############################################################################
network_train=K.models.Model([input1,input2,input3,input4],
                             [Loss1,Loss2,Loss3,Loss4])
network_train.summary()
###############################################################################
intra_data1={}
inter_data1={}
for i in range(x_test1.shape[0]):
    label_i=y_test1[i]
    intra_data1[i]=np.where(y_test1==label_i)
    inter_data1[i]=np.where(y_test1!=label_i)
intra_data2={}
inter_data2={}
for i in range(x_test2.shape[0]):
    label_i=y_test2[i]
    intra_data2[i]=np.where(y_test2==label_i)
    inter_data2[i]=np.where(y_test2!=label_i)
###############################################################################
batch_size=512
train_loss=[]
loss1=[]
loss2=[]
loss3=[]
loss4=[]
###############################################################################
iterations=1
lr=5e-3
optimizer=K.optimizers.Adam(lr=lr)
loss_weights=[1,1,1,1]
network_train.compile(optimizer=optimizer,
                      loss=[lambda y_true,y_pred: y_pred,
                            lambda y_true,y_pred: y_pred,
                            lambda y_true,y_pred: y_pred,
                            lambda y_true,y_pred: y_pred],
                            loss_weights=loss_weights)
for i in range(iterations):
    x_input1_series1_train=np.zeros(x_test1.shape)
    index0=np.zeros(x_input1_series1_train.shape[0])
    
    x_input1_series2_train=np.zeros(x_test2.shape)
    index1=np.zeros(x_input1_series2_train.shape[0])
    
    x_input2_series1_train=np.zeros(x_test1.shape)
    index2=np.zeros(x_input2_series1_train.shape[0])
    
    x_input2_series2_train=np.zeros(x_test2.shape)
    index3=np.zeros(x_input2_series2_train.shape[0])

    for ii in range(x_test1.shape[0]):
        index0[ii]=random.choice(range(x_test1.shape[0]))
        rand1=random.random()
        in_rand1=np.where(x_test1[ii,:][PCAseries1.shape[1]+PCAseries1.shape[0]:PCAseries1.shape[1]+PCAseries1.shape[0]+PCAseries2.shape[0]]>0)[0]
        out_rand1=np.where(x_test1[ii,:][PCAseries1.shape[1]+PCAseries1.shape[0]:PCAseries1.shape[1]+PCAseries1.shape[0]+PCAseries2.shape[0]]<=0)[0]
        if rand1>=0.5:
            index1[ii]=random.choice(in_rand1)
        elif rand1<0.5:
            index1[ii]=random.choice(out_rand1)
        rand2=random.random()
        if rand2>=0.5:
            index2[ii]=random.choice(intra_data1[index0[ii]][0])
        elif rand2<0.5:
            index2[ii]=random.choice(inter_data1[index0[ii]][0])
        rand3=random.random()
        if rand3>=0.5:
            index3[ii]=random.choice(intra_data2[index1[ii]][0])
        elif rand3<0.5:
            index3[ii]=random.choice(inter_data2[index1[ii]][0])

    train1=x_test1[index0.astype('int'),:]
    train2=x_test2[index1.astype('int'),:]
    train3=x_test1[index2.astype('int'),:]
    train4=x_test2[index3.astype('int'),:]

    Train=network_train.fit([train1,train2,train3,train4],
                            [np.zeros([train1.shape[0],1]),
                             np.zeros([train1.shape[0],1]),
                             np.zeros([train1.shape[0],1]),
                             np.zeros([train1.shape[0],1])],
                             batch_size=batch_size,shuffle=True)

    train_loss.append(Train.history['loss'][:][0])
    loss1.append(Train.history['loss1_loss'][:][0]*loss_weights[0])
    loss2.append(Train.history['loss2_loss'][:][0]*loss_weights[1])
    loss3.append(Train.history['loss3_loss'][:][0]*loss_weights[2])
    loss4.append(Train.history['loss4_loss'][:][0]*loss_weights[3])

    print(i,'loss=',
            Train.history['loss'][:][0],
            Train.history['loss1_loss'][:][0]*loss_weights[0],
            Train.history['loss2_loss'][:][0]*loss_weights[1],
            Train.history['loss3_loss'][:][0]*loss_weights[2],
            Train.history['loss4_loss'][:][0]*loss_weights[3])

    if i>10:
        plt.plot(train_loss[:])
        plt.plot(loss1[:])
        plt.plot(loss2[:])
        plt.plot(loss3[:])
        plt.plot(loss4[:])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.ylim(0,max(max(train_loss[len(train_loss)-10:],loss1[len(train_loss)-10:],
                                      loss2[len(train_loss)-10:],loss3[len(train_loss)-10:],
                                      loss4[len(train_loss)-10:])))
        plt.xlim(len(train_loss)-10-10,len(train_loss))
        plt.xlabel('Epoch')
        plt.legend(['Train','loss1','loss2','loss3','loss4'],loc='upper left')
        plt.show()
        
        plt.plot(train_loss[:])
        plt.plot(loss1[:])
        plt.plot(loss2[:])
        plt.plot(loss3[:])
        plt.plot(loss4[:])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train','loss1','loss2','loss3','loss4'],loc='upper left')
        plt.show()
    else:
        plt.plot(train_loss[10:])
        plt.plot(loss1[10:])
        plt.plot(loss2[10:])
        plt.plot(loss3[10:])
        plt.plot(loss4[10:])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train','loss1','loss2','loss3','loss4'],loc='upper left')
        plt.show()
############################################################################### 
network_train.load_weights('speciesAC.h5')

network_predict=K.models.Model([input1,input2,input3,input4],
                              [x1_mid1,x2_mid1,x1_mid2,x2_mid2])
[low_dim1,low_dim2,low_dim3,low_dim4]=network_predict.predict([x_test1,x_test2,x_test1,x_test2])
low_dim1=low_dim1[0:x_input1.shape[0]]
low_dim2=low_dim2[0:x_input2.shape[0]]
low_dim3=low_dim3[0:x_input1.shape[0]]
low_dim4=low_dim4[0:x_input2.shape[0]]
y_recluster_no1=recluster1[0:x_input1.shape[0]]
y_recluster_no2=recluster2[0:x_input2.shape[0]]
###############################################################################
total_recluster_type=np.concatenate([y_recluster_no1,y_recluster_no2])
###############################################################################
series1=sc.AnnData(low_dim1)
series2=sc.AnnData(low_dim2)
mergedata=series1.concatenate(series2)
mergedata.obsm['NN']=mergedata.X
sc.pp.neighbors(mergedata,n_pcs=0)
sc.tl.louvain(mergedata)
sc.tl.leiden(mergedata)
sc.tl.umap(mergedata)
sc.pl.umap(mergedata,color='louvain',size=30)
sc.pl.umap(mergedata,color='leiden',size=30)
sc.pl.umap(mergedata,color='batch',size=30)
type_louvain=mergedata.obs['louvain']
type_leiden=mergedata.obs['leiden']
type_batch=mergedata.obs['batch']
###############################################################################
umapdata=pd.DataFrame(mergedata.obsm['X_umap'].T,index=['tSNE1','tSNE2'])
umapdata1=pd.DataFrame(mergedata.obsm['X_umap'][0:PCAseries1.shape[0],:].T,index=['tSNE1','tSNE2'])
umapdata2=pd.DataFrame(mergedata.obsm['X_umap'][PCAseries1.shape[0]:,:].T,index=['tSNE1','tSNE2'])
###############################################################################
fromname='一次审核之后的结果/figure/speciesCBA_'
plot_tSNE_sepclusters(umapdata1,umapdata2,y_recluster_noSMOTE1*0,y_recluster_noSMOTE2*0+1,s=6,cluster_colors=cluster_colors,save=False,name=fromname+'label1')
plot_tSNE_sepclusters(umapdata2,umapdata1,y_recluster_noSMOTE2*0+1,y_recluster_noSMOTE1*0,s=6,cluster_colors=cluster_colors,save=False,name=fromname+'label2')
plot_tSNE_clusters(umapdata,list(map(int,np.concatenate([y_recluster_noSMOTE1*0,y_recluster_noSMOTE2*0+1]))),cluster_colors=cluster_colors,save=False, name=fromname+'batch')
plot_tSNE_clusters(umapdata,list(map(int,type_louvain)), cluster_colors=cluster_colors,save=False, name=fromname+'louvain')
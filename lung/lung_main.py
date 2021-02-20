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

we_use=[1]
     
RAWseries1=pd.read_csv('RAWlung_'+str(we_use[0])+'.csv',header=None)[1:].values.astype('single')
choose_seriestype1=pd.read_csv('reallung_'+str(we_use[0])+'.csv',header=None)[1:].values
row1=pd.read_csv('rowgenelung_'+str(we_use[0])+'.csv',header=None)[1:].values

csv_data=pd.read_csv("Lung-countsFACS.csv",header=None)
cellname=csv_data.iloc[0][1:]
csv_data=csv_data[1:]
csv_df=pd.DataFrame(csv_data)
row2=csv_df[0].values
RAWseries2=csv_df.drop(labels=0,axis=1).values.astype('int')

batch2dict=pd.read_csv('annotations_FACS.csv',header=None)[1:]
dictbatch=pd.DataFrame(batch2dict[2].values,index=batch2dict[0].values)
choose_seriestype2=[]
for i in cellname:
    if i in batch2dict[0].values:
        choose_seriestype2.append(dictbatch.loc[i][0])
    else:
        choose_seriestype2.append('0')

choose_seriestype2=np.array(choose_seriestype2)
choose_seriestype2=np.reshape(choose_seriestype2,[choose_seriestype2.shape[0],1])

cob_gene=[]
for i in row1:
    if i in row2:
        cob_gene.append(i)

line1=np.zeros(len(cob_gene))
line2=np.zeros(len(cob_gene))
index=0
for i in cob_gene:
    line1[index]=np.where(row1==i[0])[0][0]
    line2[index]=np.where(row2==i[0])[0][0]
    index+=1

RAWseries1=RAWseries1[line1.astype('int'),:]
RAWseries2=RAWseries2[line2.astype('int'),:]

fromname='lung'+str(we_use[0])

Alldata=np.concatenate([RAWseries1.T,RAWseries2.T])
Alllabel=np.concatenate([choose_seriestype1,choose_seriestype2])
Allbatch=np.concatenate([np.zeros(choose_seriestype1.shape[0]),np.zeros(choose_seriestype2.shape[0])+1])

for i in np.unique(Alllabel):
    print(i,(choose_seriestype1==i).sum(),(choose_seriestype2==i).sum())

chosen_cluster=['269',
                '268',
                '275',
                '277',
                '265',
                '287',
                '266',
                '273',
                '282',
                'B cell',
                'T cell',
                'dendritic cell',
                'endothelial cell',
                'stromal cell'
                ]

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

cluster_index2={'269':0,
                '268':1,
                '275':2,
                '277':3,
                '265':3,
                '287':3,
                '266':4,
                '273':4,
                '282':4,
                'B cell':0,
                'T cell':1,
                'dendritic cell':2,
                'endothelial cell':3,
                'stromal cell':4
                }

for i in range(Alllabel.shape[0]):
    Numlabel[i]=cluster_index2[Alllabel[i][0]]
###############################################################################
choose_seriestype1=Numlabel[Allbatch==0][Numlabel[Allbatch==0].argsort()].astype('int')
choose_seriestype2=Numlabel[Allbatch==1][Numlabel[Allbatch==1].argsort()].astype('int')
###############################################################################
min_cells=100
pca_dim=15
minnumberofcluster=10000000000
clusternumber=1
###############################################################################
anndata=sc.AnnData(pd.DataFrame(Alldata))
sc.pp.filter_genes(anndata,min_cells=min_cells)
sc.pp.normalize_per_cell(anndata,counts_per_cell_after=1e4)
sc.pp.log1p(anndata)
sc.pp.highly_variable_genes(anndata)
sc.pl.highly_variable_genes(anndata)
anndata=anndata[:,anndata.var['highly_variable']]

sc.pl.highest_expr_genes(anndata,n_top=20)
sc.tl.pca(anndata,n_comps=100,svd_solver='arpack')
sc.pl.pca(anndata)
sc.pl.pca_variance_ratio(anndata,log=True,n_pcs=100,save=[True,'pancreas'])

Alldata_aft=anndata.obsm['X_pca'][:,0:pca_dim]
Alldata_aft=preprocessing.StandardScaler().fit_transform(Alldata_aft)
Alldata_aft=preprocessing.MinMaxScaler().fit_transform(Alldata_aft)

PCAseries1=Alldata_aft[Allbatch==0,:][Numlabel[Allbatch==0].argsort()]
PCAseries2=Alldata_aft[Allbatch==1,:][Numlabel[Allbatch==1].argsort()]

choose_seriestype1=Numlabel[Allbatch==0][Numlabel[Allbatch==0].argsort()].astype('int')
choose_seriestype2=Numlabel[Allbatch==1][Numlabel[Allbatch==1].argsort()].astype('int')
###############################################################################
cluster_series1=sc.AnnData(PCAseries1)
cluster_series2=sc.AnnData(PCAseries2)

sc.pp.neighbors(cluster_series1,n_pcs=0)
sc.pp.neighbors(cluster_series2,n_pcs=0)

sc.tl.umap(cluster_series1)
sc.tl.umap(cluster_series2)

sc.tl.louvain(cluster_series1,resolution=0.5)
sc.pl.umap(cluster_series1,color='louvain',size=30)

sc.tl.louvain(cluster_series2,resolution=0.5)
sc.pl.umap(cluster_series2,color='louvain',size=30)

cluster1=np.array(list(map(int,cluster_series1.obs['louvain'])))
cluster2=np.array(list(map(int,cluster_series2.obs['louvain'])))
###############################################################################
recluster1=np.zeros(cluster1.shape[0])
recluster2=np.zeros(cluster2.shape[0])

palsecluster1=cluster1
count_cluster1=pd.value_counts(cluster_series1.obs['louvain'])
for i in range(1000000000000000):
    if count_cluster1.max()<minnumberofcluster:
        break
    else:
        print(count_cluster1.max())
        recluster1=np.zeros(cluster1.shape[0])
        recluster1_number=0  
        for i in np.unique(palsecluster1):
            index=palsecluster1==i
            if index.sum()<minnumberofcluster:
                thisrecluster=np.zeros(index.sum())
                recluster1[index]=thisrecluster+recluster1_number
                recluster1_number=len(np.unique(recluster1))
            else:
                data=PCAseries1[index]
                anndata=sc.AnnData(data)
                sc.pp.neighbors(anndata,n_pcs=0)
                sc.tl.louvain(anndata)
                thisrecluster=np.array(list(map(int,anndata.obs['louvain'])))
                recluster1[index]=thisrecluster+recluster1_number
                recluster1_number=len(np.unique(recluster1))
        palsecluster1=recluster1.astype('int')
        count_cluster1=pd.value_counts(palsecluster1)
    
palsecluster2=cluster2
count_cluster2=pd.value_counts(cluster_series2.obs['louvain'])
for i in range(1000000000000000):
    if count_cluster2.max()<minnumberofcluster:
        break
    else:
        print(count_cluster2.max())
        recluster2=np.zeros(cluster2.shape[0])
        recluster2_number=0  
        for i in np.unique(palsecluster2):
            index=palsecluster2==i
            if index.sum()<minnumberofcluster:
                thisrecluster=np.zeros(index.sum())
                recluster2[index]=thisrecluster+recluster2_number
                recluster2_number=len(np.unique(recluster2))
            else:
                data=PCAseries2[index]
                anndata=sc.AnnData(data)
                sc.pp.neighbors(anndata,n_pcs=0)
                sc.tl.louvain(anndata)
                thisrecluster=np.array(list(map(int,anndata.obs['louvain'])))
                recluster2[index]=thisrecluster+recluster2_number
                recluster2_number=len(np.unique(recluster2))
        palsecluster2=recluster2.astype('int')
        count_cluster2=pd.value_counts(palsecluster2)
recluster1=palsecluster1
recluster2=palsecluster2
###############################################################################
series1=sc.AnnData(PCAseries1)
series2=sc.AnnData(PCAseries2)
sc.pp.neighbors(series1,n_pcs=0)
sc.pp.neighbors(series2,n_pcs=0)
sc.tl.umap(series1)
sc.tl.umap(series2)
df1=pd.DataFrame(choose_seriestype1)
df1=pd.Series(np.reshape(df1.values,df1.values.shape[0]), dtype="category")
series1.obs['real']=df1.values
df2=pd.DataFrame(choose_seriestype2)
df2=pd.Series(np.reshape(df2.values,df2.values.shape[0]), dtype="category")
series2.obs['real']=df2.values
sc.pl.umap(series1,color='real',size=30)
sc.pl.umap(series2,color='real',size=30)
df1=pd.DataFrame(recluster1.astype('int'))
df1=pd.Series(np.reshape(df1.values,df1.values.shape[0]), dtype="category")
series1.obs['recluster']=df1.values
df2=pd.DataFrame(recluster2.astype('int'))
df2=pd.Series(np.reshape(df2.values,df2.values.shape[0]), dtype="category")
series2.obs['recluster']=df2.values
sc.pl.umap(series1,color='recluster',size=30)
sc.pl.umap(series2,color='recluster',size=30)
###############################################################################
def dis(P,Q,distance_method):
    if distance_method==0:
        return np.sqrt(np.sum(np.square(P-Q)))
    if distance_method==1:
        return 1-(np.multiply(P,Q).sum()/(np.sqrt(np.sum(np.square(P)))*np.sqrt(np.sum(np.square(Q)))))
###############################################################################
if len(np.unique(recluster1))<=len(np.unique(recluster2)):
    a=PCAseries1
    PCAseries1=PCAseries2
    PCAseries2=a
    b=choose_seriestype1
    choose_seriestype1=choose_seriestype2
    choose_seriestype2=b
    c=cluster1
    cluster1=cluster2
    cluster2=c
    d=recluster1
    recluster1=recluster2
    recluster2=d
###############################################################################
correlation_recluster=np.zeros([len(np.unique(recluster1)),len(np.unique(recluster2))])
correlation_recluster_cell=np.zeros([recluster1.shape[0],recluster2.shape[0]])

for i in range(len(np.unique(recluster1))):
    for j in range(len(np.unique(recluster2))):
        print(i,j)
        index_series1=np.where(recluster1==i)[0]
        index_series2=np.where(recluster2==j)[0]
        cell_series1=PCAseries1[index_series1,:]
        cell_series2=PCAseries2[index_series2,:]
        mean1=0
        for iq in range(cell_series1.shape[0]):
            for jq in range(cell_series2.shape[0]):
                mean1+=dis(cell_series1[iq,:],cell_series2[jq,:],1)
        correlation_recluster[i,j]=mean1/(cell_series1.shape[0]*cell_series2.shape[0])
        for ii in range(cell_series1.shape[0]):
            for jj in range(cell_series2.shape[0]):
                mean2=dis(cell_series1[ii,:],cell_series2[jj,:],0)
                correlation_recluster_cell[index_series1[ii],index_series2[jj]]=mean2

plt.imshow(correlation_recluster)
plt.imshow(correlation_recluster_cell)
correlation_recluster_div=-np.log10(correlation_recluster)
correlation_recluster_cell_div=-np.log10(correlation_recluster_cell)
correlation_recluster_norm=(correlation_recluster_div-correlation_recluster_div.min())/(correlation_recluster_div.max()-correlation_recluster_div.min())
correlation_recluster_cell_norm=(correlation_recluster_cell_div-correlation_recluster_cell_div.min())/(correlation_recluster_cell_div.max()-correlation_recluster_cell_div.min())
plt.imshow(correlation_recluster_norm)
plt.imshow(correlation_recluster_cell_norm)
###############################################################################
correlation_recluster_select=np.zeros(correlation_recluster_norm.shape)

recluster_mid=np.zeros(recluster1.shape)
for kk in range(correlation_recluster_norm.shape[0]):
    ind=np.sort(correlation_recluster_norm[kk,:])
    select=correlation_recluster_norm[kk,:]<ind[-clusternumber]
    select=(select==False)
    recluster_mid[recluster1==kk]+=int(np.where(select==True)[0])
    correlation_recluster_select[kk,:]=correlation_recluster_norm[kk,:]*select
plt.imshow(correlation_recluster_select)
correlation_recluster_cell_final=correlation_recluster_cell*0
for i in range(correlation_recluster_cell_norm.shape[0]):
    for j in range(correlation_recluster_cell_norm.shape[1]):
        label1=recluster1[i]
        label2=recluster2[j]
        mean1=correlation_recluster_select[label1,label2]
        mean2=correlation_recluster_cell_norm[i,j]
        if mean1==0:
            correlation_recluster_cell_final[i,j]=0
        else:
            correlation_recluster_cell_final[i,j]=mean2
plt.imshow(correlation_recluster_select)
plt.imshow(correlation_recluster_cell_final)
recluster1=recluster_mid.astype('int')
sort_correlation_recluster_cell_final=correlation_recluster_cell_final[recluster1.argsort(),:]
sort_correlation_recluster_cell_final=sort_correlation_recluster_cell_final[:,recluster2.argsort()]
###############################################################################
heatmap(correlation_recluster_cell_final,choose_seriestype1,choose_seriestype2,save=False,name='pancreasmatrix')
################################################################################
x_input1=np.zeros([PCAseries1.shape[0],PCAseries1.shape[1]+PCAseries1.shape[0]+PCAseries2.shape[0]+recluster2.max()+1])
x_input2=np.zeros([PCAseries2.shape[0],PCAseries2.shape[1]+PCAseries2.shape[0]+PCAseries1.shape[0]+recluster2.max()+1])
for i in range(PCAseries1.shape[0]):
    print(i)
    x_input1[i,0:PCAseries1.shape[1]]=PCAseries1[i,:]
    x_input1[i,PCAseries1.shape[1]:PCAseries1.shape[1]+PCAseries1.shape[0]]=K.utils.np_utils.to_categorical(i,PCAseries1.shape[0])
    x_input1[i,PCAseries1.shape[1]+PCAseries1.shape[0]:PCAseries1.shape[1]+PCAseries1.shape[0]+PCAseries2.shape[0]]=correlation_recluster_cell_final[i,:]
    x_input1[i,PCAseries1.shape[1]+PCAseries1.shape[0]+PCAseries2.shape[0]:]=K.utils.np_utils.to_categorical(recluster1[i],recluster2.max()+1)
for j in range(PCAseries2.shape[0]):
    print(j)
    x_input2[j,0:PCAseries2.shape[1]]=PCAseries2[j,:]
    x_input2[j,PCAseries2.shape[1]:PCAseries2.shape[1]+PCAseries2.shape[0]]=K.utils.np_utils.to_categorical(j,PCAseries2.shape[0])
    x_input2[j,PCAseries2.shape[1]+PCAseries2.shape[0]:PCAseries2.shape[1]+PCAseries2.shape[0]+PCAseries1.shape[0]]=correlation_recluster_cell_final[:,j]
    x_input2[j,PCAseries2.shape[1]+PCAseries2.shape[0]+PCAseries1.shape[0]:]=K.utils.np_utils.to_categorical(recluster2[j],recluster2.max()+1)
###############################################################################
x_input1_new=x_input1
recluster1_new=recluster1
x_input2_new=x_input2
recluster2_new=recluster2
###############################################################################
if x_input1_new.shape[0]>=x_input2_new.shape[0]:
    x_test1=x_input1_new
    y_test1=recluster1_new
    y_testreal1=choose_seriestype1
    repeat_num=int(np.ceil(x_input1_new.shape[0]/x_input2_new.shape[0])) 
    x_test2=np.tile(x_input2_new,(repeat_num,1))
    y_test2=np.tile(recluster2_new,repeat_num)
    y_testreal2=np.tile(choose_seriestype2,repeat_num)
    x_test2=x_test2[0:x_test1.shape[0],:]
    y_test2=y_test2[0:x_test1.shape[0]]
    y_testreal2=y_testreal2[0:x_test1.shape[0]]
elif x_input1_new.shape[0]<x_input2_new.shape[0]:
    x_test2=x_input2_new
    y_test2=recluster2_new     
    y_testreal2=choose_seriestype2
    repeat_num=int(np.ceil(x_input2_new.shape[0]/x_input1_new.shape[0]))
    x_test1=np.tile(x_input1_new,(repeat_num,1))
    y_test1=np.tile(recluster1_new,repeat_num)
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
layer2=layer

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

x1=layers.Dense(layer2,activation=activation)(Data1)
x2=layers.Dense(layer2,activation=activation)(Data2)
x3=layers.Dense(layer2,activation=activation)(Data3)
x4=layers.Dense(layer2,activation=activation)(Data4)

x1=layers.BatchNormalization()(x1)
x2=layers.BatchNormalization()(x2)
x3=layers.BatchNormalization()(x3)
x4=layers.BatchNormalization()(x4)

x1_mid1=layers.Dense(layer2,activation=activation)(layers.concatenate([x1,x2]))
x2_mid1=layers.Dense(layer2,activation=activation)(layers.concatenate([x1,x2]))
x1_mid2=layers.Dense(layer2,activation=activation)(layers.concatenate([x3,x4]))
x2_mid2=layers.Dense(layer2,activation=activation)(layers.concatenate([x3,x4]))

x1_mid1=layers.BatchNormalization()(x1_mid1)
x2_mid1=layers.BatchNormalization()(x2_mid1)
x1_mid2=layers.BatchNormalization()(x1_mid2)
x2_mid2=layers.BatchNormalization()(x2_mid2)

layer_classify=layers.Dense(recluster2_new.max()+1,activation='relu')

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

#loss1
AE_loss_1=Lambda(MSE)([output1,x_concat1])
AE_loss_2=Lambda(MSE)([output2,x_concat2])
#loss2
cls_loss_1=Lambda(MSE)([y1,Relabel1])
cls_loss_2=Lambda(MSE)([y2,Relabel2])
cls_loss_3=Lambda(MSE)([y3,Relabel3])
cls_loss_4=Lambda(MSE)([y4,Relabel4])
#loss3
interweight1=Lambda(loss_weight)([Index1,Cor2])
interweight4=Lambda(loss_weight)([Index3,Cor4])
interloss_1=Lambda(MSE)([x1_mid1,x2_mid1])
interloss_4=Lambda(MSE)([x1_mid2,x2_mid2])
interloss_1=layers.Multiply()([interweight1,interloss_1])
interloss_4=layers.Multiply()([interweight4,interloss_4])
#loss4
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
batch_size=128
train_loss=[]
loss1=[]
loss2=[]
loss3=[]
loss4=[]
###############################################################################
iterations=1000000000
lr=1e-3
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

    if i>100:
        plt.plot(train_loss[:])
        plt.plot(loss1[:])
        plt.plot(loss2[:])
        plt.plot(loss3[:])
        plt.plot(loss4[:])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.ylim(0,max(max(train_loss[i-100:],loss1[i-100:],loss2[i-100:],loss3[i-100:],loss4[i-100:])))
        plt.xlim(i-100,i)
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
        plt.plot(train_loss[100:])
        plt.plot(loss1[100:])
        plt.plot(loss2[100:])
        plt.plot(loss3[100:])
        plt.plot(loss4[100:])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train','loss1','loss2','loss3','loss4'],loc='upper left')
        plt.show()
############################################################################### 
network_train.load_weights('lungweight.h5')        

network_predict=K.models.Model([input1,input2,input3,input4],
                              [x1_mid1,x2_mid1,x1_mid2,x2_mid2]) 
[low_dim1,low_dim2,low_dim3,low_dim4]=network_predict.predict([x_test1,x_test2,x_test1,x_test2])
low_dim1=low_dim1[0:x_input1.shape[0]]
low_dim2=low_dim2[0:x_input2.shape[0]]
low_dim3=low_dim3[0:x_input1.shape[0]]
low_dim4=low_dim4[0:x_input2.shape[0]]
y_real_no1=y_testreal1[0:x_input1.shape[0]]
y_recluster_no1=recluster1[0:x_input1.shape[0]]
y_real_no2=y_testreal2[0:x_input2.shape[0]]
y_recluster_no2=recluster2[0:x_input2.shape[0]]
total_real_type=np.concatenate([y_real_no1,y_real_no2])
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
df=pd.DataFrame(total_real_type.astype('int'))
df=pd.Series(np.reshape(df.values,df.values.shape[0]), dtype="category")
mergedata.obs['real']=df.values
sc.pl.umap(mergedata,color='louvain',size=30)
sc.pl.umap(mergedata,color='leiden',size=30)
sc.pl.umap(mergedata,color='batch',size=30)
sc.pl.umap(mergedata,color='real',size=30)
type_louvain=mergedata.obs['louvain']
type_leiden=mergedata.obs['leiden']
type_batch=mergedata.obs['batch']
type_real=mergedata.obs['real']
###############################################################################
umapdata=pd.DataFrame(mergedata.obsm['X_umap'].T,index=['tSNE1','tSNE2'])
umapdata1=pd.DataFrame(mergedata.obsm['X_umap'][0:PCAseries1.shape[0],:].T,index=['tSNE1','tSNE2'])
umapdata2=pd.DataFrame(mergedata.obsm['X_umap'][PCAseries1.shape[0]:,:].T,index=['tSNE1','tSNE2'])
###############################################################################
plot_tSNE_batchclusters(umapdata1,umapdata2,choose_seriestype1,choose_seriestype2,s=6,cluster_colors=cluster_colors,save=False,name=fromname+'batch1')
plot_tSNE_batchclusters(umapdata2,umapdata1,choose_seriestype2,choose_seriestype1,s=6,cluster_colors=cluster_colors,save=False,name=fromname+'batch2')
plot_tSNE_clusters(umapdata,list(map(int,type_batch)), cluster_colors=cluster_colors,save=False,name=fromname+'batch')
plot_tSNE_sepclusters(umapdata1,umapdata2,choose_seriestype1,choose_seriestype2,s=6,cluster_colors=cluster_colors,save=False,name=fromname+'label1')
plot_tSNE_sepclusters(umapdata2,umapdata1,choose_seriestype2,choose_seriestype1,s=6,cluster_colors=cluster_colors,save=False,name=fromname+'label2')
plot_tSNE_clusters(umapdata,list(map(int,type_real)), cluster_colors=cluster_colors,save=False, name=fromname+'label')
#sio.savemat('lung_ourdata.mat',{'mergedata':mergedata.X,'umapdata':umapdata.values})
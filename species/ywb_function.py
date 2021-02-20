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
from collections import Counter
import matplotlib.pyplot as plt 
from keras.regularizers import l2
from sklearn import preprocessing
from keras.layers.core import Lambda
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from mpl_toolkits.axes_grid1 import make_axes_locatable

def color(value):
  digit = list(map(str, range(10))) + list("ABCDEF")
  if isinstance(value, tuple):
    string = '#'
    for i in value:
      a1 = i // 16
      a2 = i % 16
      string += digit[a1] + digit[a2]
    return string
  elif isinstance(value, str):
    a1 = digit.index(value[1]) * 16 + digit.index(value[2])
    a2 = digit.index(value[3]) * 16 + digit.index(value[4])
    a3 = digit.index(value[5]) * 16 + digit.index(value[6])
    return (a1, a2, a3)

cluster_colors=[
        color((213,94,0)),
        color((0,114,178)),
        color((204,121,167)),
        color((0,158,115)),
        color((86,180,233)),
        color((230,159,0)),
        color((240,228,66)),
        color((0,0,0)),
        '#D3D3D3',
        '#FF00FF',
        
        
        '#aec470',
        '#b3ee3d',
        '#de4726',
        '#f69149',
        '#f81919',     
        '#ff49b0',
        '#f05556',
        '#fadf0b', 
        '#f8c495',
        '#ffc1c1',
        '#ffc125',
        '#ffc0cb',
        '#ffbbff',
        '#ffb90f',
        '#ffb6c1',
        '#ffb5c5',
        '#ff83fa',
        '#ff8c00',
        '#ff4040',
        '#ff3030',
        '#ff34b3',
        '#00fa9a',
        '#ca4479',
        '#eead0e',
        '#ff1493',
        '#0ab4e4',
        '#1e6a87', 
        '#800080',
        '#00e5ee',
        '#c71585',
        '#027fd0', 
        '#004dba', 
        '#0a9fb4',
        '#004b71', 
        '#285528',
        '#2f7449',
        '#21b183',
        '#3e4198',
        '#4e14a6',   
        '#5dd73d',   
        '#64a44e', 
        '#6787d6',
        '#6c6b6b',
        '#6c6b6b',    
        '#7759a4',     
        '#78edff',
        '#762a14',
        '#9805cc',  
        '#9b067d',
        '#af7efe',
        '#a7623d']
    
def plot_tSNE_clusters(df_tSNE,labels,cluster_colors=None,s=6,save=False,name=None):
    fig,ax=plt.subplots(figsize=(4, 4))
    ax.scatter(df_tSNE.loc['tSNE1'], df_tSNE.loc['tSNE2'],s=s,alpha=0.8,lw=0,c=[cluster_colors[i] for i in labels])
    ax.axis('equal')
    ax.set_axis_off()
    if save==True:
        plt.savefig('{}.eps'.format(name),dpi=600,format='eps')
def plot_tSNE_batchclusters(df_tSNE1,df_tSNE2,labels1,labels2,cluster_colors=None,s=0.8,save=False,name=None):
    fig,ax=plt.subplots(figsize=(4, 4))
    ax.scatter(df_tSNE2.loc['tSNE1'], df_tSNE2.loc['tSNE2'],s=s,alpha=0.8,lw=0,c='#D3D3D3')
    ax.scatter(df_tSNE1.loc['tSNE1'], df_tSNE1.loc['tSNE2'],s=s,alpha=0.8,lw=0,c=[cluster_colors[1] for i in labels1])
    ax.axis('equal')
    ax.set_axis_off()
    if save==True:
        plt.savefig('{}.eps'.format(name),dpi=600,format='eps')
def plot_tSNE_sepclusters(df_tSNE1,df_tSNE2,labels1,labels2,cluster_colors=None,s=0.8,save=False,name=None):
    fig,ax=plt.subplots(figsize=(4, 4))
    ax.scatter(df_tSNE2.loc['tSNE1'], df_tSNE2.loc['tSNE2'],s=s,alpha=0.8,lw=0,c='#D3D3D3')
    ax.scatter(df_tSNE1.loc['tSNE1'], df_tSNE1.loc['tSNE2'],s=s,alpha=0.8,lw=0,c=[cluster_colors[i] for i in labels1])
    ax.axis('equal')
    ax.set_axis_off()
    if save==True:
        plt.savefig('{}.eps'.format(name),dpi=600,format='eps')
def plot_tSNE_cluster(df_tSNE,labels,cluster_colors=None,s=6,save=False,name=None):     
    index=[[] for i in range(np.max(labels)+1)]
    for i in range(len(labels)):
        index[int(labels[i])].append(i)
    index=[i for i in index if i!=[]]
    
    for i in range(len(np.unique(labels))):
        color=np.array(labels)[index[i]][0]
        fig,ax=plt.subplots()
        ax.scatter(df_tSNE.loc['tSNE1'], df_tSNE.loc['tSNE2'],c='#D3D3D3',s=s,lw=0)
        ax.scatter(df_tSNE.loc['tSNE1'].iloc[index[i]],df_tSNE.loc['tSNE2'].iloc[index[i]],c=[cluster_colors[k] for k in np.array(labels)[index[i]]],s=s,lw=0)
        ax.axis('equal')
        ax.set_axis_off()
        if save == True:
            plt.savefig('{}.eps'.format(name+str(color)), dpi=600,format='eps')
def gen_labels(df, model):
    if str(type(model)).startswith("<class 'sklearn.cluster"):
        cell_labels = dict(zip(df.columns, model.labels_))
        label_cells = {}
        for l in np.unique(model.labels_):
            label_cells[l] = []
        for i, label in enumerate(model.labels_):
            label_cells[label].append(df.columns[i])
        cellID = list(df.columns)
        labels = list(model.labels_)
        labels_a = model.labels_
    elif type(model) == np.ndarray:
        cell_labels = dict(zip(df.columns, model))
        label_cells = {}
        for l in np.unique(model):
            label_cells[l] = []
        for i, label in enumerate(model):
            label_cells[label].append(df.columns[i])
        cellID = list(df.columns)
        labels = list(model)
        labels_a = model
    else:
        print('Error wrong input type')
    return cell_labels, label_cells, cellID, labels, labels_a
def heatmap(correlation_recluster_cell_final,choose_seriestype1,choose_seriestype2,save=False,name=''):
    df=pd.DataFrame(correlation_recluster_cell_final)
    labels1=np.array(choose_seriestype1)
    labels2=np.array(choose_seriestype2)
    cell_labels1,label_cells1,cellID1,labels1,labels_a1=gen_labels(df.T,np.array(labels1)) 
    cell_labels2,label_cells2,cellID2,labels2,labels_a2=gen_labels(df,np.array(labels2)) 
    
    optimal_order=np.unique(np.concatenate([labels1,labels2]))
    cl,lc=gen_labels(df,np.array(labels2))[:2]
    
    optimal_sort_cells=sum([lc[i] for i in np.unique(labels2)],[])
    optimal_sort_labels=[cl[i] for i in optimal_sort_cells]
    fig,axHM=plt.subplots(figsize=(9,5))
    df_full=df.copy()
    z=df_full.values
    z=pd.DataFrame(z, index=df_full.index,columns=df_full.columns)
    z=z.loc[:,optimal_sort_cells].values
    
    im=axHM.pcolormesh(z,cmap='viridis',vmax=1)
    plt.gca().invert_yaxis()
    plt.xlim(xmax=len(labels2))
    plt.xticks([])
    plt.yticks([])
    
    divider=make_axes_locatable(axHM)
    axLabel1=divider.append_axes("top",.3,pad=0,sharex=axHM)
    axLabel2=divider.append_axes("left",.3,pad=0,sharex=axHM)
    
    counter2=Counter(labels2)
    counter1=Counter(labels1)
    
    pos2=0
    pos1=0
    for l in optimal_order:
        axLabel1.barh(y=0,left=pos2,width=counter2[l],color=cluster_colors[l],linewidth=0.5,edgecolor=cluster_colors[l])
        pos2+=counter2[l]
    optimal_order=np.flipud(optimal_order)
    for l in optimal_order:
        axLabel2.bar(x=0,bottom=pos1,height=counter1[l],color=cluster_colors[l],linewidth=50,edgecolor=cluster_colors[l])
        pos1+=counter1[l]
    
    axLabel1.set_xlim(xmax=len(labels2))
    axLabel1.axis('off')
    axLabel2.set_ylim(ymax=len(labels1))
    axLabel2.axis('off')
    
    cax=fig.add_axes([.91,0.13,0.01,0.22])
    colorbar=fig.colorbar(im,cax=cax,ticks=[0,1])
    colorbar.set_ticklabels(['0','max'])
    
    plt.savefig('{}.jpg'.format(name),dpi=600,format='jpg')
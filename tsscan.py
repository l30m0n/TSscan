"""
TSSCAN: Density based spatial clustering on time series data
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from tqdm import tqdm
from sklearn.cluster import DBSCAN

class TSscan():
    """Perform DBSCAN over time-series data

    Parameters
    --------------

    eps : float, default=0.1
        The maximum distance beetween 2 points for being considered as neighbours

    min_samples : int, default=2 
        The number of samples (or total weight) in the neighborhood for a point to be considered as a core point. 
        (This include the point itself)

    dist : str, or callable, default='euclidean'
        metric to be used in the computation of dynamic time warping. For other metrics search in scipy.spatial.distance

    dist_rescale : bool, default=True
        Transform the dynamic time warping over two time-series to be the mean between the distance of the points associated


    Attributes
    --------------
    dist: callable,
        used distance between 2 time-series points
    

    model : callable,
        DBScan model

    divergence_matrix : ndarray
        dynamic time warping matrix computed between all time series

    labels_ : ndarray
        Cluster labels for each points in the dataset given by fit

    



    """


    def __init__(self, eps=0.1, min_samples=2, dist=euclidean, dist_rescale=True):
        self.dist=dist

        self.dist_rescale=True #set the use of path len in divergence matrix computation

        self.model=DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        
        ### computed after 

        self.divergence_matrix=None
        self.data_=None
        self.labels_=None
        

    def fit(self,data):
        """Perform clustering from features time-series refrequenced
        Parameters
        --------------
        data : ndarray of shape (nsample, nfeatures)

        Return
        -------
        self : object
            Returns a fitted instance of self
        """
        if self.data_==None or self.data_!=data:
            self.data_=data

            divergence_matrix=np.zeros((data.shape[0],data.shape[0]))

            len_path_matrix=np.ones((data.shape[0],data.shape[0]))

            for i in range(data.shape[0]):

                for j in range(i+1,data.shape[0]):

                    distance, path=fastdtw(data[i],data[j])

                    divergence_matrix[i,j]=distance

                    len_path_matrix[i,j]=len(path)

            if self.dist_rescale:

                divergence_matrix=divergence_matrix/len_path_matrix

            self.divergence_matrix=divergence_matrix+divergence_matrix.T

        self.model.fit(self.divergence_matrix)
        
        self.labels_=self.model.labels_
        

    def fit_predict(self,data):
        """Perform clustering from features time-series refrequenced

        Parameters
        --------------
        data : ndarray of shape (nsample, nfeatures)

        Return
        -------
        self : object
            Returns a fitted instance of self
        
        label_ :  ndarray
            Cluster labels for each points in the dataset given by fit
        """
        self.fit(data)

        return self.labels_

    def cluster_medoid(self,idx_cluster):
        """Perform a search of medoïd in our fitted model cluster given

        Parameters
        --------------
        idx_cluster : int
            id of cluster chosen
        
        Return
        --------------
        medoid : ndarray,
            medoid element of the cluster
        id_medoid : int
            id on data_array of medeid of the cluster
        """

        idx_list=[]
        for lab,nb in zip(self.labels_,range(len(self.labels_.tolist()))):
            if lab==idx_cluster:
                idx_list.append(nb)
        metric_cluster=pd.DataFrame(self.divergence_matrix)[idx_list].T[idx_list]
        metric_cluster=metric_cluster.sum()
        metric_cluster=metric_cluster.sort_values(ascending=True)
        val=metric_cluster.index[0]
        return self.data_[val],val

    def mean_cluster(self,idx_cluster,get_len=False):
        """Perform a search of dynamic based averaging in our fitted model cluster given

        Parameters
        --------------
        idx_cluster : int
            id of cluster chosen
        
        get_len : bool, default=False
            if true return len of elements
        
        Return
        --------------
        mean : ndarray,
            medoid element of the cluster

        len : list
            return len of elements
        """
        idx_list=[]
        moy_,val=self.cluster_medoid(idx_cluster)
        labels=self.labels_.tolist()
        labels.pop(val)
        data=self.data_.tolist()
        data.pop(val)
        for lab,series in zip(labels,data):
            if lab==idx_cluster:
                idx_list.append(series)  

        idx_list=np.array(idx_list)  
        nb_values=1
        map_values=list(map(lambda i:[i],moy_))
        for i in idx_list:
            dist, path=distance, path=fastdtw(moy_,i,dist=euclidean)
            for j in path:
                map_values[j[0]].append(i[j[1]])
        mean_val=list(map(lambda i:np.array(pd.DataFrame(np.array(i)).mean()),map_values)) 
        len_val=list(map(lambda i:len(i),map_values)) 
        if get_len:
            return len_val
        return mean_val

    

    def cluster_plot(self,plot_mean=True):
        """Graphic représentation of each fitted clusters in 1 dimensionnal data

        Parameters
        --------------
        plot_mean : Bool
            add mean barycenter of dtw on the plot ofn each clusters

        Return
        --------------       
        plot : matplotlib.subplot
            subplot of eache clusterised data over the time
        """
        max_val=max(self.labels_.tolist())

        fig, axes=plt.subplots(1, int(max_val)+2, sharex=True, sharey=True)

        ax=axes.ravel()

        for i,lab in zip(range(self.data_.shape[0]),self.labels_):

            ax[lab+1].plot([a for a in range(len(self.data_[i].tolist()))],self.data_[i].tolist())

        ax[0].set_title("Noise")

        for i in range(1,int(max_val) + 2):

            ax[i].set_title(f"Cluster {i}")

        if plot_mean:
            labels=set(self.labels_)
            try : 
                labels.remove(-1)
            except:
                pass
            for lab in labels:
                mean_current=self.mean_cluster(lab)
                ax[lab+1].plot([a for a in range(len(mean_current))],mean_current,color="black",linewidth=2)

        plt.show()

    
    def metric_plot(self):
        """Plot distance matrix between all time-series

        Return
        -------------
        dist_matrix : seaborn.heatmap
            plot of heatmap
        """
        data_grouped=pd.DataFrame()
        data_grouped['labels']=self.labels_.tolist()
        A=self.divergence_matrix.copy()
        A=pd.DataFrame(A).sort_values(data_grouped.index.to_list())
        A=A.T
        A=pd.DataFrame(A).sort_values(data_grouped.index.to_list())
        sns.heatmap(A)

        
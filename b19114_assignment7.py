"""
Name : Shashwat Mishra
Roll no. : B19114
Mobile no. : 6387481964

"""
#importing numpy,pandas and matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading the training and test data 
train=pd.read_csv('mnist-tsne-train.csv',usecols=['dimention 1','dimension 2'])
test=pd.read_csv('mnist-tsne-test.csv',usecols=['dimention 1','dimention 2'])
train=train.rename(columns={'dimention 1':'dimension 1'})
test=test.rename(columns={'dimention 1':'dimension 1','dimention 2':'dimension 2'})
#reading the training and test label
train_label=pd.read_csv('mnist-tsne-train.csv',usecols=['labels'])
test_label=pd.read_csv('mnist-tsne-test.csv',usecols=['labels'])

#function to calculate the purity score given true value and predicted value
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
def purity_score(y_true, y_pred):
 # compute contingency matrix (also called confusion matrix)
 contingency_matrix=metrics.cluster.contingency_matrix(y_true, y_pred)
 #print(contingency_matrix)
 # Find optimal one-to-one mapping between cluster labels and true labels
 row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
 # Return cluster accuracy
 return contingency_matrix[row_ind,col_ind].sum()/np.sum(contingency_matrix)

#function for scatter plot of the  clusters
def Plot(data,centers,prediction,title):
    plt.scatter(data['dimension 1'][:], data['dimension 2'][:],c=prediction,cmap='nipy_spectral',marker='.')
    plt.colorbar().ax.set_xlabel('Cluster')
    if centers.any():
        plt.scatter(centers.T[0],centers.T[1],marker="*",s=100,color='k')
    plt.xlabel('dimension 2')
    plt.ylabel('dimension 1')
    plt.title(title)
    plt.show()
    


print('\n------------------------------------------------')
#function for Kmeans clustering 
from sklearn.cluster import KMeans
def Kmeans(k,data):
    kmeans = KMeans(n_clusters=k,random_state=42)
    kmeans.fit(train)
    kmeans_prediction = list(kmeans.predict(data))
    centers=kmeans.cluster_centers_
    Plot(data,centers,kmeans_prediction,'KMeans clustering with K='+str(k))
    return kmeans_prediction
    
k = 10
#Kmeans on training data
print('\nKMeans clustering on Training Data:')
#part A 
kmeans_prediction_train = Kmeans(k,train)
#part B
print('Purity score =',purity_score(train_label,kmeans_prediction_train))
#Kmeans on test data
print('\nKMeans clustering on Test Data:')
#part C
kmeans_prediction_test = Kmeans(k,test)
#part D
print('Purity score =',purity_score(test_label,kmeans_prediction_test))



print('\n---------------------------------------------------------')
#function for clustering using GMM
from sklearn.mixture import GaussianMixture 
def GMM(k,data):
    gmm = GaussianMixture(n_components = k,random_state=42)
    gmm.fit(train)
    GMM_prediction = gmm.predict(data)
    centers=gmm.means_
    Plot(data,centers,GMM_prediction,'GMM clustering with K='+str(k))
    return GMM_prediction
    
    
k=10
#GMM on trianing data
print('\nGMM clustering on Training Data:')
#part A
GMM_prediction_train = GMM(k,train)
#part B
print('Purity score =',purity_score(train_label,GMM_prediction_train))
#GMM on test data
print('\nGMM clustering on Test Data:')
#part C
GMM_prediction_test = GMM(k,test)
#part D
print('Purity score =',purity_score(test_label,GMM_prediction_test))


print('\n----------------------------------------------------------')
from sklearn.cluster import DBSCAN
from scipy import spatial as spatial
#function for DBSCAN given the value of epsilon and minpoints
def dbscan(eps, minsamples):
    dbscan_model=DBSCAN(eps=eps, min_samples=minsamples).fit(train)
    DBSCAN_predictions = dbscan_model.labels_
    #for training data
    #part A
    print('\nDBSCAN clustering on Training Data:')
    Plot(train,np.array([False]),DBSCAN_predictions,'DBSCAN clustering with eps='+str(eps)+' min sample='+str(minsamples))
    #part B
    print('Purity score =',purity_score(train_label,DBSCAN_predictions))
    #for test data
    #function to assign test points to clusters using DBSCAN
    def dbscan_predict(dbscan_model, X_new, metric= spatial.distance.euclidean):
        y_new = np.ones(shape=len(X_new), dtype=int)*-1 
        for j, x_new in enumerate(X_new):
            for i, x_core in enumerate(dbscan_model.components_):
                if metric(x_new, x_core) < dbscan_model.eps:
                    y_new[j] =dbscan_model.labels_[dbscan_model.core_sample_indices_[i]]
                    break
        return y_new
    dbtest = dbscan_predict(dbscan_model,test.values, metric = spatial.distance.euclidean)
    #part C
    print('\nDBSCAN clustering on Test Data:')
    Plot(test,np.array([False]),dbtest,'DBSCAN clustering with eps='+str(eps)+' min sample='+str(minsamples))
    #part D
    print('Purity score =',purity_score(test_label,dbtest))
#calling the function defined above
dbscan(5,10)

print('\n-----------------------------------------------------------------------')
#part 
#function to calculate distortion measure in KMeans clustering 
def destortion(k):
    kmeans = KMeans(n_clusters=k,random_state=42)
    kmeans.fit(train)
    return kmeans.inertia_
#function to calculate log likelihood in GMM
def loglikelihood(k):
    gmm = GaussianMixture(n_components = k,random_state=42)
    gmm.fit(train)
    return gmm.lower_bound_


#doing Kmeans for different values of k
k=[2,5,8,12,18,20]
dist=[]
for i in k:
    print('\nk = ',i)
    #Kmeans on training data
    print('\nKMeans clustering on Training Data:')
    kmeans_prediction_train = Kmeans(i,train)
    print('Purity score =',purity_score(train_label,kmeans_prediction_train))
    #Kmeans on test data
    print('\nKMeans clustering on Test Data:')
    kmeans_prediction_test = Kmeans(i,test)
    print('Purity score =',purity_score(test_label,kmeans_prediction_test))
    dist.append(destortion(i))
#plotting distortion measure versus k
plt.plot(k,dist,marker = 'o')
plt.xlabel('k')
plt.ylabel('distortion measure')
plt.title('elbow method for Kmeans clustering')
plt.show()
print('Optimum number of clusters using elbow method for K-means clustering=',8)


#doing GMM for different values of k
totallog=[]
for i in k:
    print('\nk = ',i)
    #GMM on trianing data
    print('\nGMM clustering on Training Data:')
    GMM_prediction_train = GMM(i,train)
    print('Purity score =',purity_score(train_label,GMM_prediction_train))
    #GMM on test data
    print('\nGMM clustering on Test Data:')
    GMM_prediction_test = GMM(i,test)
    print('Purity score =',purity_score(test_label,GMM_prediction_test))
    totallog.append(loglikelihood(i))
#plotting log likelihood for different values of k    
plt.plot(k,totallog,marker = 'o')
plt.xlabel('k')
plt.ylabel('log likelihood')
plt.title('elbow method for GMM')
plt.show()
print('Optimum number of clusters using elbow method for GMM clustering=',8)
    

#part B
#varying the value of epsilon and minpoints in DBSCAN
epsilon=[1,5,10]
minpoints=[1,10,30,50]
for i in epsilon:
    dbscan(i,10)   
for i in minpoints:
    dbscan(5,i)




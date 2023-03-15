#importing required datasets
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

#datasets
dataset=pd.read_csv("Mall_Customers.csv")
X=dataset.iloc[:, :].values
#print(X)

#dendrogram section

print("Loading Dendrogram...Please Wait")
dendrogram=sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

#hierarchical clustering

clustering=AgglomerativeClustering(n_clusters=5)
y_hc=clustering.fit_predict(X)
#print(y_hc)
plt.scatter(X[y_hc==0 , 0], X[y_hc==0 , 1],c='red', label='cluster 1')
plt.scatter(X[y_hc==1 , 0], X[y_hc==1 , 1],c='green', label='cluster 2')
plt.scatter(X[y_hc==2 , 0], X[y_hc==2 , 1],c='blue', label='cluster 3')
plt.scatter(X[y_hc==3 , 0], X[y_hc==3 , 1],c='orange', label='cluster 4')
plt.scatter(X[y_hc==4 , 0], X[y_hc==4 , 1],c='black', label='cluster 5')
plt.title("Cluster of Customers")
plt.xlabel("Annual Income(K $)")
plt.ylabel("Spending Score(1-100)")
plt.legend()
plt.show()
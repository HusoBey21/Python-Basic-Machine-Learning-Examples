import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_biclusters
from sklearn.datasets import samples_generator as sg
from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn.metrics  import consensus_score
data,rows,columns=make_biclusters(shape=(300,300),n_clusters=5,noise=5,shuffle=False,random_state=0)
data,row_idx,col_idx=sg._shuffle(data,random_state=0)
plt.matshow(data,cmap=plt.cm.Blues)
plt.title("Gerçek veri kümesi")
model=SpectralCoclustering(n_clusters=5,random_state=0)
model.fit(data)
score=consensus_score(model.biclusters_,(rows[:,row_idx],columns[:,col_idx]))
print("consensus score:{:.3f}".format(score))
fit_data=data[np.argsort(model.row_labels_)]
fit_data=fit_data[:,np.argsort(model.column_labels_)]
plt.matshow(fit_data,cmap=plt.cm.Blues)
plt.title("İkili kümelemeden sonra:İkili kümelemeyi göster")
plt.show()

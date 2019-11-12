import time
import numpy as np
from distutils.version import LooseVersion
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import skimage
from skimage.data import coins
from skimage.transform import rescale
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering
#Bunlar sk görüntüsü içinde tanımlanır
if LooseVersion(skimage.__version__) >= '0.14':
    rescale_params={'anti_aliasing':False,'multichannel':False}
else:
    rescale_params={}
orig_coins=coins()
#Bir numpy dizisine madeni paraları yükle
#Gerçek işleme hız boyutunu yüzde 20 olarak yeniden boyutlandır.
smoothened_coins=gaussian_filter(orig_coins,sigma=2)
rescaled_coins=rescale(smoothened_coins,0.2,mode='reflect',**rescale_params)
graph=image.img_to_graph(rescaled_coins)
beta=10
eps=1e-6
graph.data=np.exp(-beta*graph.data/graph.data.std()) +eps
N_REGIONS=25
for assign_labels in {'kmeans','discretize'}:
    t0=time.time()
    labels=spectral_clustering(graph,n_clusters=N_REGIONS,assign_labels=assign_labels,random_state=42)
    t1=time.time()
    labels=labels.reshape(rescaled_coins.shape)
    plt.figure(figsize=(5,5))
    plt.imshow(rescaled_coins,cmap=plt.cm.gray)
    for l in range(N_REGIONS):
        plt.contour(labels==1,colors=[plt.cm.nipy_spectral(1/float(N_REGIONS))])
    plt.xticks(())
    plt.yticks(())
    title='Hayali kümeleme:%s,%.2f s' % (assign_labels,(t1 - t0))
    print(title)
    plt.title(title)
    

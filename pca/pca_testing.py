from NN_tuning import create_X
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import matplotlib.cm as cm
# https://stackoverflow.com/questions/12236566/setting-different-color-for-each-series-in-scatter-plot-on-matplotlib

evCA = np.array(loadmat('evCA.mat')['evCA']).ravel()
X = create_X(200)[0]
sc = StandardScaler()
pca = PCA(n_components = 2)
data = pca.fit_transform(X)
print(pca.explained_variance_ratio_)
print(pca.components_)
colors = cm.rainbow(np.linspace(0, 1, 167))
m=0
for i in range(0,33400,200):
    plt.scatter(data[i:i+200,0], data[i:i+200,1], color = colors[m])
    m += 1
plt.show()

data3 = pca.fit_transform(X[1200:33400-1200])
print(pca.explained_variance_ratio_)
print(pca.components_)
colors = cm.rainbow(np.linspace(0, 1, 31))
m=0
for i in range(0,31000,1000):
    plt.scatter(data3[i:i+200,0], data3[i:i+200,1], color = colors[m])
    m += 1
plt.show()

data4 = pca.fit_transform(X)
print(pca.explained_variance_ratio_)
print(pca.components_)
colors = cm.rainbow(np.linspace(0, 1, 35))
m=0
for i in range(0,1200,600):
    plt.scatter(data4[i:i+600,0], data4[i:i+600,1], color = colors[m])
    m += 1
for i in range(1200,32200,1000):
    plt.scatter(data4[i:i+1000,0], data4[i:i+1000,1], color = colors[m])
    m += 1
for i in range(32200,33400,600):
    plt.scatter(data4[i:i+600,0], data4[i:i+600,1], color = colors[m])
    m += 1
plt.show()
colors = cm.rainbow(np.linspace(0, 1, 3))
m=0
for i in range(3200,30200,9000):
    plt.scatter(data4[i:i+9000,0], data4[i:i+9000,1], color = colors[m])
    m += 1
plt.show()

scaled = sc.fit_transform(X)
data2 = pca.fit_transform(scaled)
print(pca.explained_variance_ratio_)
print(pca.components_)
colors = cm.rainbow(np.linspace(0, 1, 167))
m=0
for i in range(0,33400,200):
    plt.scatter(data2[i:i+200,0], data2[i:i+200,1], color = colors[m])
    m += 1
plt.show()


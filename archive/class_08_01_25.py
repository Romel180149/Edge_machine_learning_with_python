from sklearn.cluster import KMeans
import numpy as np

X = np.random.randint(0, 50, (20, 2))
print(X)
import matplotlib.pyplot as plt
plt.scatter(X[:,1],X[:,1])
plt.show()
z = np.array([3,3])
plt.predict(z)
# array([3]),dtype = int32
c = np.array([3], dtype=np.int32)
print(c)
plt.cluster_center
# adjusted rand score 
from sklearn.metrics import adjusted_rand_score
print(adjusted_rand_score([1],))


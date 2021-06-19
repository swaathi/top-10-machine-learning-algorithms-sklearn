from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from base import Base

import pylab as pl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


df = Base.clean()
X = df.drop(['Survived'], axis=1)
y = df['Survived']
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, random_state=0)

model = KNeighborsClassifier(n_neighbors=7)
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

print("\n\nK-Nearest Neighbor Accuracy Score:", Base.accuracy_score(ytest, ypred), "%")

# Transforming n-features into 2
pca = PCA(n_components=2).fit(X)
pca_2d = pca.transform(X)

for i in range(0, pca_2d.shape[0]):
    if y[i] == 1:
        c1 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='g', marker='o')
    elif y[i] == 0:
        c2 = pl.scatter(pca_2d[i,0], pca_2d[i,1], c='r', marker='+')

pl.legend([c1, c2], ['Survived', 'Deceased'])
pl.title('Titanic Survivors')
plt.savefig('visualizations/knn.png')

from sklearn.neighbors import KNeighborsClassifier
from base import Base

Xtrain, Xtest, ytrain, ytest = Base.clean_and_split()

model = KNeighborsClassifier(n_neighbors=1)
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

print("\n\nK-Nearest Neighbor Accuracy Score:", Base.accuracy_score(ytest, ypred), "%")

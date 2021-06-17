from sklearn.naive_bayes import GaussianNB
from base import Base

Xtrain, Xtest, ytrain, ytest = Base.clean_and_split()

model = GaussianNB()
model.fit(Xtrain, ytrain)
ypred = model.predict(Xtest)

print("\n\nNaive Bayes Accuracy Score:", Base.accuracy_score(ytest, ypred), "%")
